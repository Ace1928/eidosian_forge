from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def GetIssuesHelper(self, entity):
    """Get issues in a conversion worksapce."""
    entity_issues = []
    for issue in entity.issues:
        if issue.severity in self.high_severity_issues:
            entity_issues.append({'parentEntity': entity.parentEntity, 'shortName': entity.shortName, 'entityType': six.text_type(entity.entityType).replace('DATABASE_ENTITY_TYPE_', ''), 'issueType': six.text_type(issue.type).replace('ISSUE_TYPE_', ''), 'issueSeverity': six.text_type(issue.severity).replace('ISSUE_SEVERITY_', ''), 'issueCode': issue.code, 'issueMessage': issue.message})
    return entity_issues