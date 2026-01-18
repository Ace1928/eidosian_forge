from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
class DynamodbWaiterFactory(BaseWaiterFactory):

    def __init__(self, module):
        client = module.client('dynamodb')
        super().__init__(module, client)

    @property
    def _waiter_model_data(self):
        data = super()._waiter_model_data
        ddb_data = dict(table_exists=dict(operation='DescribeTable', delay=20, maxAttempts=25, acceptors=[dict(expected='ACTIVE', matcher='path', state='success', argument='Table.TableStatus'), dict(expected='ResourceNotFoundException', matcher='error', state='retry')]), table_not_exists=dict(operation='DescribeTable', delay=20, maxAttempts=25, acceptors=[dict(expected='ResourceNotFoundException', matcher='error', state='success')]), global_indexes_active=dict(operation='DescribeTable', delay=20, maxAttempts=25, acceptors=[dict(expected='ResourceNotFoundException', matcher='error', state='failure'), dict(expected=False, matcher='path', state='success', argument='contains(keys(Table), `GlobalSecondaryIndexes`)'), dict(expected='ACTIVE', matcher='pathAll', state='success', argument='Table.GlobalSecondaryIndexes[].IndexStatus'), dict(expected='CREATING', matcher='pathAny', state='retry', argument='Table.GlobalSecondaryIndexes[].IndexStatus'), dict(expected='UPDATING', matcher='pathAny', state='retry', argument='Table.GlobalSecondaryIndexes[].IndexStatus'), dict(expected='DELETING', matcher='pathAny', state='retry', argument='Table.GlobalSecondaryIndexes[].IndexStatus'), dict(expected=True, matcher='path', state='success', argument='length(Table.GlobalSecondaryIndexes) == `0`')]))
        data.update(ddb_data)
        return data