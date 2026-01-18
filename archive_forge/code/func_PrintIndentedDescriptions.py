import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
def PrintIndentedDescriptions(printer, ls, name, prefix=''):
    if ls:
        with printer.Indent(indent=prefix):
            with printer.CommentContext():
                width = printer.CalculateWidth() - len(prefix)
                printer()
                printer(name + ':')
                for x in ls:
                    description = '%s: %s' % (x.name, x.description)
                    for line in textwrap.wrap(description, width, initial_indent='  ', subsequent_indent='    '):
                        printer(line)