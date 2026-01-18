from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
class ManyToManyDP(DependencyProcessor):

    def per_property_dependencies(self, uow, parent_saves, child_saves, parent_deletes, child_deletes, after_save, before_delete):
        uow.dependencies.update([(parent_saves, after_save), (child_saves, after_save), (after_save, child_deletes), (before_delete, parent_saves), (before_delete, parent_deletes), (before_delete, child_deletes), (before_delete, child_saves)])

    def per_state_dependencies(self, uow, save_parent, delete_parent, child_action, after_save, before_delete, isdelete, childisdelete):
        if not isdelete:
            if childisdelete:
                uow.dependencies.update([(save_parent, after_save), (after_save, child_action)])
            else:
                uow.dependencies.update([(save_parent, after_save), (child_action, after_save)])
        else:
            uow.dependencies.update([(before_delete, child_action), (before_delete, delete_parent)])

    def presort_deletes(self, uowcommit, states):
        if not self.passive_deletes:
            for state in states:
                uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)

    def presort_saves(self, uowcommit, states):
        if not self.passive_updates:
            for state in states:
                if self._pks_changed(uowcommit, state):
                    history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_OFF)
        if not self.cascade.delete_orphan:
            return
        for state in states:
            history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_NO_INITIALIZE)
            if history:
                for child in history.deleted:
                    if self.hasparent(child) is False:
                        uowcommit.register_object(child, isdelete=True, operation='delete', prop=self.prop)
                        for c, m, st_, dct_ in self.mapper.cascade_iterator('delete', child):
                            uowcommit.register_object(st_, isdelete=True)

    def process_deletes(self, uowcommit, states):
        secondary_delete = []
        secondary_insert = []
        secondary_update = []
        processed = self._get_reversed_processed_set(uowcommit)
        tmp = set()
        for state in states:
            history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
            if history:
                for child in history.non_added():
                    if child is None or (processed is not None and (state, child) in processed):
                        continue
                    associationrow = {}
                    if not self._synchronize(state, child, associationrow, False, uowcommit, 'delete'):
                        continue
                    secondary_delete.append(associationrow)
                tmp.update(((c, state) for c in history.non_added()))
        if processed is not None:
            processed.update(tmp)
        self._run_crud(uowcommit, secondary_insert, secondary_update, secondary_delete)

    def process_saves(self, uowcommit, states):
        secondary_delete = []
        secondary_insert = []
        secondary_update = []
        processed = self._get_reversed_processed_set(uowcommit)
        tmp = set()
        for state in states:
            need_cascade_pks = not self.passive_updates and self._pks_changed(uowcommit, state)
            if need_cascade_pks:
                passive = attributes.PASSIVE_OFF | attributes.INCLUDE_PENDING_MUTATIONS
            else:
                passive = attributes.PASSIVE_NO_INITIALIZE | attributes.INCLUDE_PENDING_MUTATIONS
            history = uowcommit.get_attribute_history(state, self.key, passive)
            if history:
                for child in history.added:
                    if processed is not None and (state, child) in processed:
                        continue
                    associationrow = {}
                    if not self._synchronize(state, child, associationrow, False, uowcommit, 'add'):
                        continue
                    secondary_insert.append(associationrow)
                for child in history.deleted:
                    if processed is not None and (state, child) in processed:
                        continue
                    associationrow = {}
                    if not self._synchronize(state, child, associationrow, False, uowcommit, 'delete'):
                        continue
                    secondary_delete.append(associationrow)
                tmp.update(((c, state) for c in history.added + history.deleted))
                if need_cascade_pks:
                    for child in history.unchanged:
                        associationrow = {}
                        sync.update(state, self.parent, associationrow, 'old_', self.prop.synchronize_pairs)
                        sync.update(child, self.mapper, associationrow, 'old_', self.prop.secondary_synchronize_pairs)
                        secondary_update.append(associationrow)
        if processed is not None:
            processed.update(tmp)
        self._run_crud(uowcommit, secondary_insert, secondary_update, secondary_delete)

    def _run_crud(self, uowcommit, secondary_insert, secondary_update, secondary_delete):
        connection = uowcommit.transaction.connection(self.mapper)
        if secondary_delete:
            associationrow = secondary_delete[0]
            statement = self.secondary.delete().where(sql.and_(*[c == sql.bindparam(c.key, type_=c.type) for c in self.secondary.c if c.key in associationrow]))
            result = connection.execute(statement, secondary_delete)
            if result.supports_sane_multi_rowcount() and result.rowcount != len(secondary_delete):
                raise exc.StaleDataError("DELETE statement on table '%s' expected to delete %d row(s); Only %d were matched." % (self.secondary.description, len(secondary_delete), result.rowcount))
        if secondary_update:
            associationrow = secondary_update[0]
            statement = self.secondary.update().where(sql.and_(*[c == sql.bindparam('old_' + c.key, type_=c.type) for c in self.secondary.c if c.key in associationrow]))
            result = connection.execute(statement, secondary_update)
            if result.supports_sane_multi_rowcount() and result.rowcount != len(secondary_update):
                raise exc.StaleDataError("UPDATE statement on table '%s' expected to update %d row(s); Only %d were matched." % (self.secondary.description, len(secondary_update), result.rowcount))
        if secondary_insert:
            statement = self.secondary.insert()
            connection.execute(statement, secondary_insert)

    def _synchronize(self, state, child, associationrow, clearkeys, uowcommit, operation):
        self._verify_canload(child)
        if child is None:
            return False
        if child is not None and (not uowcommit.session._contains_state(child)):
            if not child.deleted:
                util.warn("Object of type %s not in session, %s operation along '%s' won't proceed" % (mapperutil.state_class_str(child), operation, self.prop))
            return False
        sync.populate_dict(state, self.parent, associationrow, self.prop.synchronize_pairs)
        sync.populate_dict(child, self.mapper, associationrow, self.prop.secondary_synchronize_pairs)
        return True

    def _pks_changed(self, uowcommit, state):
        return sync.source_modified(uowcommit, state, self.parent, self.prop.synchronize_pairs)