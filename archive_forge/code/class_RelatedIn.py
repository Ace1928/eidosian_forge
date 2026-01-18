from django.db.models.lookups import (
class RelatedIn(In):

    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource):
            if self.rhs_is_direct_value():
                self.rhs = [get_normalized_value(val, self.lhs)[0] for val in self.rhs]
                if hasattr(self.lhs.output_field, 'path_infos'):
                    target_field = self.lhs.output_field.path_infos[-1].target_fields[-1]
                    self.rhs = [target_field.get_prep_value(v) for v in self.rhs]
            elif not getattr(self.rhs, 'has_select_fields', True) and (not getattr(self.lhs.field.target_field, 'primary_key', False)):
                if getattr(self.lhs.output_field, 'primary_key', False) and self.lhs.output_field.model == self.rhs.model:
                    target_field = self.lhs.field.name
                else:
                    target_field = self.lhs.field.target_field.name
                self.rhs.set_values([target_field])
        return super().get_prep_lookup()

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            from django.db.models.sql.where import AND, OR, SubqueryConstraint, WhereNode
            root_constraint = WhereNode(connector=OR)
            if self.rhs_is_direct_value():
                values = [get_normalized_value(value, self.lhs) for value in self.rhs]
                for value in values:
                    value_constraint = WhereNode()
                    for source, target, val in zip(self.lhs.sources, self.lhs.targets, value):
                        lookup_class = target.get_lookup('exact')
                        lookup = lookup_class(target.get_col(self.lhs.alias, source), val)
                        value_constraint.add(lookup, AND)
                    root_constraint.add(value_constraint, OR)
            else:
                root_constraint.add(SubqueryConstraint(self.lhs.alias, [target.column for target in self.lhs.targets], [source.name for source in self.lhs.sources], self.rhs), AND)
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)