from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddValidationGroup(parser, verb):
    """Adds a --validate-only or --force flag to the given parser."""
    validation_group = parser.add_group(mutex=True)
    validation_group.add_argument('--validate-only', help='Only validate the stream, but do not %s any resources.\n      The default is false.' % verb.lower(), action='store_true', default=False)
    validation_group.add_argument('--force', help='%s the stream without validating it.' % verb, action='store_true', default=False)