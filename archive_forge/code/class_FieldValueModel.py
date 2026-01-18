from tempest.lib.cli import output_parser
class FieldValueModel(Model):
    """This converts cli output from messy lists/dicts to neat attributes."""

    def __init__(self, out):
        """This parses output with fields and values like:

            +----------------+------------------------------+
            | Field          | Value                        |
            +----------------+------------------------------+
            | action         | CREATE                       |
            | created_at     | 2015-08-20T17:22:17.000000   |
            | description    | None                         |
            +----------------+------------------------------+

        These are then accessible as:

            model.action
            model.created_at
            model.description

        """
        table = output_parser.table(out)
        value_lines = []
        prev_field = None
        for field, value in table['values']:
            if field == '':
                value_lines.append(value)
                setattr(self, prev_field, '\n'.join(value_lines))
            else:
                setattr(self, field, value)
                prev_field = field
                value_lines = [value]