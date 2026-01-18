from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub.util import InvalidArgumentError
class SchemasClient(object):
    """Client for schemas service in the Cloud Pub/Sub API."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule(client)
        self._service = self.client.projects_schemas

    def Commit(self, schema_ref, schema_definition, schema_type):
        """Commits a revision for a Schema.

    Args:
      schema_ref: The full schema_path.
      schema_definition: The new schema definition to commit.
      schema_type: The type of the schema (avro or protocol-buffer).

    Returns:
    Schema: the committed Schema revision
    """
        schema = self.messages.Schema(name=schema_ref, type=ParseSchemaType(self.messages, schema_type), definition=schema_definition)
        commit_req = self.messages.PubsubProjectsSchemasCommitRequest(commitSchemaRequest=self.messages.CommitSchemaRequest(schema=schema), name=schema_ref)
        return self._service.Commit(commit_req)

    def Rollback(self, schema_ref, revision_id):
        """Rolls back to a previous schema revision.

    Args:
      schema_ref: The path of the schema to rollback.
      revision_id: The revision_id to rollback to.

    Returns:
    Schema: the new schema revision you have rolled back to.

    Raises:
      InvalidArgumentError: If no revision_id is provided.
    """
        rollback_req = self.messages.PubsubProjectsSchemasRollbackRequest(rollbackSchemaRequest=self.messages.RollbackSchemaRequest(revisionId=revision_id), name=schema_ref)
        return self._service.Rollback(rollback_req)

    def DeleteRevision(self, schema_ref):
        """Deletes a schema revision.

    Args:
      schema_ref: The path of the schema, with the revision_id.

    Returns:
    Schema: the deleted schema revision.
    """
        if not CheckRevisionIdInSchemaPath(schema_ref):
            raise NoRevisionIdSpecified()
        delete_revision_req = self.messages.PubsubProjectsSchemasDeleteRevisionRequest(name=schema_ref)
        return self._service.DeleteRevision(delete_revision_req)