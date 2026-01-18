from copy import deepcopy
class ForeignKeyName(TableColumns):
    """Hold a reference to a foreign key name."""

    def __init__(self, from_table, from_columns, to_table, to_columns, suffix_template, create_fk_name):
        self.to_reference = TableColumns(to_table, to_columns)
        self.suffix_template = suffix_template
        self.create_fk_name = create_fk_name
        super().__init__(from_table, from_columns)

    def references_table(self, table):
        return super().references_table(table) or self.to_reference.references_table(table)

    def references_column(self, table, column):
        return super().references_column(table, column) or self.to_reference.references_column(table, column)

    def rename_table_references(self, old_table, new_table):
        super().rename_table_references(old_table, new_table)
        self.to_reference.rename_table_references(old_table, new_table)

    def rename_column_references(self, table, old_column, new_column):
        super().rename_column_references(table, old_column, new_column)
        self.to_reference.rename_column_references(table, old_column, new_column)

    def __str__(self):
        suffix = self.suffix_template % {'to_table': self.to_reference.table, 'to_column': self.to_reference.columns[0]}
        return self.create_fk_name(self.table, self.columns, suffix)