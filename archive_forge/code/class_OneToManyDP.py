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
class OneToManyDP(DependencyProcessor):

    def per_property_dependencies(self, uow, parent_saves, child_saves, parent_deletes, child_deletes, after_save, before_delete):
        if self.post_update:
            child_post_updates = unitofwork.PostUpdateAll(uow, self.mapper.primary_base_mapper, False)
            child_pre_updates = unitofwork.PostUpdateAll(uow, self.mapper.primary_base_mapper, True)
            uow.dependencies.update([(child_saves, after_save), (parent_saves, after_save), (after_save, child_post_updates), (before_delete, child_pre_updates), (child_pre_updates, parent_deletes), (child_pre_updates, child_deletes)])
        else:
            uow.dependencies.update([(parent_saves, after_save), (after_save, child_saves), (after_save, child_deletes), (child_saves, parent_deletes), (child_deletes, parent_deletes), (before_delete, child_saves), (before_delete, child_deletes)])

    def per_state_dependencies(self, uow, save_parent, delete_parent, child_action, after_save, before_delete, isdelete, childisdelete):
        if self.post_update:
            child_post_updates = unitofwork.PostUpdateAll(uow, self.mapper.primary_base_mapper, False)
            child_pre_updates = unitofwork.PostUpdateAll(uow, self.mapper.primary_base_mapper, True)
            if not isdelete:
                if childisdelete:
                    uow.dependencies.update([(child_action, after_save), (after_save, child_post_updates)])
                else:
                    uow.dependencies.update([(save_parent, after_save), (child_action, after_save), (after_save, child_post_updates)])
            elif childisdelete:
                uow.dependencies.update([(before_delete, child_pre_updates), (child_pre_updates, delete_parent)])
            else:
                uow.dependencies.update([(before_delete, child_pre_updates), (child_pre_updates, delete_parent)])
        elif not isdelete:
            uow.dependencies.update([(save_parent, after_save), (after_save, child_action), (save_parent, child_action)])
        else:
            uow.dependencies.update([(before_delete, child_action), (child_action, delete_parent)])

    def presort_deletes(self, uowcommit, states):
        should_null_fks = not self.cascade.delete and (not self.passive_deletes == 'all')
        for state in states:
            history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
            if history:
                for child in history.deleted:
                    if child is not None and self.hasparent(child) is False:
                        if self.cascade.delete_orphan:
                            uowcommit.register_object(child, isdelete=True)
                        else:
                            uowcommit.register_object(child)
                if should_null_fks:
                    for child in history.unchanged:
                        if child is not None:
                            uowcommit.register_object(child, operation='delete', prop=self.prop)

    def presort_saves(self, uowcommit, states):
        children_added = uowcommit.memo(('children_added', self), set)
        should_null_fks = not self.cascade.delete_orphan and (not self.passive_deletes == 'all')
        for state in states:
            pks_changed = self._pks_changed(uowcommit, state)
            if not pks_changed or self.passive_updates:
                passive = attributes.PASSIVE_NO_INITIALIZE | attributes.INCLUDE_PENDING_MUTATIONS
            else:
                passive = attributes.PASSIVE_OFF | attributes.INCLUDE_PENDING_MUTATIONS
            history = uowcommit.get_attribute_history(state, self.key, passive)
            if history:
                for child in history.added:
                    if child is not None:
                        uowcommit.register_object(child, cancel_delete=True, operation='add', prop=self.prop)
                children_added.update(history.added)
                for child in history.deleted:
                    if not self.cascade.delete_orphan:
                        if should_null_fks:
                            uowcommit.register_object(child, isdelete=False, operation='delete', prop=self.prop)
                    elif self.hasparent(child) is False:
                        uowcommit.register_object(child, isdelete=True, operation='delete', prop=self.prop)
                        for c, m, st_, dct_ in self.mapper.cascade_iterator('delete', child):
                            uowcommit.register_object(st_, isdelete=True)
            if pks_changed:
                if history:
                    for child in history.unchanged:
                        if child is not None:
                            uowcommit.register_object(child, False, self.passive_updates, operation='pk change', prop=self.prop)

    def process_deletes(self, uowcommit, states):
        if self.post_update or not self.passive_deletes == 'all':
            children_added = uowcommit.memo(('children_added', self), set)
            for state in states:
                history = uowcommit.get_attribute_history(state, self.key, self._passive_delete_flag)
                if history:
                    for child in history.deleted:
                        if child is not None and self.hasparent(child) is False:
                            self._synchronize(state, child, None, True, uowcommit, False)
                            if self.post_update and child:
                                self._post_update(child, uowcommit, [state])
                    if self.post_update or not self.cascade.delete:
                        for child in set(history.unchanged).difference(children_added):
                            if child is not None:
                                self._synchronize(state, child, None, True, uowcommit, False)
                                if self.post_update and child:
                                    self._post_update(child, uowcommit, [state])

    def process_saves(self, uowcommit, states):
        should_null_fks = not self.cascade.delete_orphan and (not self.passive_deletes == 'all')
        for state in states:
            history = uowcommit.get_attribute_history(state, self.key, attributes.PASSIVE_NO_INITIALIZE)
            if history:
                for child in history.added:
                    self._synchronize(state, child, None, False, uowcommit, False)
                    if child is not None and self.post_update:
                        self._post_update(child, uowcommit, [state])
                for child in history.deleted:
                    if should_null_fks and (not self.cascade.delete_orphan) and (not self.hasparent(child)):
                        self._synchronize(state, child, None, True, uowcommit, False)
                if self._pks_changed(uowcommit, state):
                    for child in history.unchanged:
                        self._synchronize(state, child, None, False, uowcommit, True)

    def _synchronize(self, state, child, associationrow, clearkeys, uowcommit, pks_changed):
        source = state
        dest = child
        self._verify_canload(child)
        if dest is None or (not self.post_update and uowcommit.is_deleted(dest)):
            return
        if clearkeys:
            sync.clear(dest, self.mapper, self.prop.synchronize_pairs)
        else:
            sync.populate(source, self.parent, dest, self.mapper, self.prop.synchronize_pairs, uowcommit, self.passive_updates and pks_changed)

    def _pks_changed(self, uowcommit, state):
        return sync.source_modified(uowcommit, state, self.parent, self.prop.synchronize_pairs)