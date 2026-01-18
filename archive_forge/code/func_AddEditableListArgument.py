from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddEditableListArgument(parser, singular, plural, category_help, add_metavar=None, remove_metavar=None, clear_arg=None, clear_help=None, collector_type=None, dict_like=False, dest=None):
    """Adds arguments to `parser` for modifying a list field.

  A generic implementation of the style guidelines at
  go/gcloud-style#createupdate-command-flags.

  Args:
    parser: the ArgumentParser to which the arguments will be added.
    singular: singular form of the name of the field to be modified.
    plural: singular form of the name of the field to be modified.
    category_help: help text for the commands as a whole. Should explain what
      the field itself is.
    add_metavar: text to use as a placeholder in the add argument.
    remove_metavar: text to use as a placeholder in the remove argument.
    clear_arg: what to name the argument that clears the list.
    clear_help: help text for the argument that clears the list.
    collector_type: type for the add and remove arguments.
    dict_like: whether the list field has keys and values.
    dest: suffix for fields in the parsed argument object.
  """
    mutex_group = parser.add_mutually_exclusive_group()
    add_remove_group = mutex_group.add_argument_group(help=category_help)
    add_remove_group.add_argument('--add-' + singular.lower().replace(' ', '-'), action=arg_parsers.UpdateAction if dict_like else 'append', type=collector_type or (arg_parsers.ArgDict() if dict_like else arg_parsers.ArgList()), dest='add_' + dest if dest else None, help='Adds a new %s to the set of %s.' % (singular, plural), metavar=add_metavar or singular.upper().replace(' ', '-'))
    add_remove_group.add_argument('--remove-' + singular.lower().replace(' ', '-'), action='append', type=collector_type or arg_parsers.ArgList(), dest='remove_' + dest if dest else None, help='Removes an existing %s from the set of %s.' % (singular, plural), metavar=remove_metavar or singular.upper().replace(' ', '-'))
    mutex_group.add_argument(clear_arg if clear_arg else '--clear-' + plural.lower().replace(' ', '-'), action='store_true', dest='clear_' + dest if dest else None, help=clear_help if clear_help else 'Removes all %s.' % plural)