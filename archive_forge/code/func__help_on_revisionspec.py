import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def _help_on_revisionspec(name):
    """Generate the help for revision specs."""
    import re
    import breezy.revisionspec
    out = []
    out.append('Revision Identifiers\n\nA revision identifier refers to a specific state of a branch\'s history.  It\ncan be expressed in several ways.  It can begin with a keyword to\nunambiguously specify a given lookup type; some examples are \'last:1\',\n\'before:yesterday\' and \'submit:\'.\n\nAlternately, it can be given without a keyword, in which case it will be\nchecked as a revision number, a tag, a revision id, a date specification, or a\nbranch specification, in that order.  For example, \'date:today\' could be\nwritten as simply \'today\', though if you have a tag called \'today\' that will\nbe found first.\n\nIf \'REV1\' and \'REV2\' are revision identifiers, then \'REV1..REV2\' denotes a\nrevision range. Examples: \'3647..3649\', \'date:yesterday..-1\' and\n\'branch:/path/to/branch1/..branch:/branch2\' (note that there are no quotes or\nspaces around the \'..\').\n\nRanges are interpreted differently by different commands. To the "log" command,\na range is a sequence of log messages, but to the "diff" command, the range\ndenotes a change between revisions (and not a sequence of changes).  In\naddition, "log" considers a closed range whereas "diff" and "merge" consider it\nto be open-ended, that is, they include one end but not the other.  For example:\n"brz log -r 3647..3649" shows the messages of revisions 3647, 3648 and 3649,\nwhile "brz diff -r 3647..3649" includes the changes done in revisions 3648 and\n3649, but not 3647.\n\nThe keywords used as revision selection methods are the following:\n')
    details = []
    details.append('\nIn addition, plugins can provide other keywords.')
    details.append('\nA detailed description of each keyword is given below.\n')
    indent_re = re.compile('^    ', re.MULTILINE)
    for prefix, i in breezy.revisionspec.revspec_registry.iteritems():
        doc = i.help_txt
        if doc == breezy.revisionspec.RevisionSpec.help_txt:
            summary = 'N/A'
            doc = summary + '\n'
        else:
            summary, doc = doc.split('\n', 1)
            while doc[-2:] == '\n\n' or doc[-1:] == ' ':
                doc = doc[:-1]
        out.append(':{}\n\t{}'.format(i.prefix, summary))
        details.append(':{}\n{}'.format(i.prefix, doc))
    return '\n'.join(out + details)