import abc
from itertools import compress
import stevedore
from . import command
def _generate_columns_and_selector(self, parsed_args, column_names):
    """Generate included columns and selector according to parsed args.

        :param parsed_args: argparse.Namespace instance with argument values
        :param column_names: sequence of strings containing names
                             of output columns
        """
    if not parsed_args.columns:
        columns_to_include = column_names
        selector = None
    else:
        columns_to_include = [c for c in column_names if c in parsed_args.columns]
        if not columns_to_include:
            raise ValueError('No recognized column names in %s. Recognized columns are %s.' % (str(parsed_args.columns), str(column_names)))
        selector = [c in columns_to_include for c in column_names]
    return (columns_to_include, selector)