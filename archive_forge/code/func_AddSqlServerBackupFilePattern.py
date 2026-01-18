from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddSqlServerBackupFilePattern(parser):
    """Adds a --sqlserver-backup-file-pattern flag to the given parser."""
    help_text = 'Pattern that describes the default backup naming strategy. The specified pattern should ensure lexicographical order of backups. The pattern should define the following capture group set\nepoch - unix timestamp\nExample: For backup files TestDB.1691448240.bak, TestDB.1691448254.trn, TestDB.1691448284.trn.final use pattern: .*\\.(<epoch>\\d*)\\.(trn|bak|trn\\.final) or .*\\.(<timestamp>\\d*)\\.(trn|bak|trn\\.final)'
    parser.add_argument('--sqlserver-backup-file-pattern', help=help_text, default='.*(\\.|_)(<epoch>\\d*)\\.(trn|bak|trn\\.final)')