from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
class CfnParameters(parameters.Parameters):
    PSEUDO_PARAMETERS = PARAM_STACK_ID, PARAM_STACK_NAME, PARAM_REGION = ('AWS::StackId', 'AWS::StackName', 'AWS::Region')

    def _pseudo_parameters(self, stack_identifier):
        stack_id = stack_identifier.arn() if stack_identifier is not None else 'None'
        stack_name = stack_identifier and stack_identifier.stack_name
        yield parameters.Parameter(self.PARAM_STACK_ID, parameters.Schema(parameters.Schema.STRING, _('Stack ID'), default=str(stack_id)))
        if stack_name:
            yield parameters.Parameter(self.PARAM_STACK_NAME, parameters.Schema(parameters.Schema.STRING, _('Stack Name'), default=stack_name))
            yield parameters.Parameter(self.PARAM_REGION, parameters.Schema(parameters.Schema.STRING, default='ap-southeast-1', constraints=[constr.AllowedValues(['us-east-1', 'us-west-1', 'us-west-2', 'sa-east-1', 'eu-west-1', 'ap-southeast-1', 'ap-northeast-1'])]))