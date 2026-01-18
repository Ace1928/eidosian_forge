from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.backup_restore import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ReadSubstitutionRuleFile(file_arg):
    """Reads content of the substitution rule file specified in file_arg."""
    if not file_arg:
        return messages.FieldList(messages.StringField(number=1, repeated=True), [])
    log.warning('The substitutionRules field is deprecated and can only be managed via gcloud/API. Please migrate to transformation rules.')
    data = console_io.ReadFromFileOrStdin(file_arg, binary=False)
    ms = api_util.GetMessagesModule()
    temp_restore_config = export_util.Import(message_type=ms.RestoreConfig, stream=data, schema_path=export_util.GetSchemaPath('gkebackup', 'v1', 'SubstitutionRules'))
    return temp_restore_config.substitutionRules