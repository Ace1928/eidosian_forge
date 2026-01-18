from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FunctionDeclaration(_messages.Message):
    """Structured representation of a function declaration as defined by the
  [OpenAPI 3.0 specification](https://spec.openapis.org/oas/v3.0.3). Included
  in this declaration are the function name and parameters. This
  FunctionDeclaration is a representation of a block of code that can be used
  as a `Tool` by the model and executed by the client.

  Fields:
    description: Optional. Description and purpose of the function. Model uses
      it to decide how and whether to call the function.
    name: Required. The name of the function to call. Must start with a letter
      or an underscore. Must be a-z, A-Z, 0-9, or contain underscores, dots
      and dashes, with a maximum length of 64.
    parameters: Optional. Describes the parameters to this function in JSON
      Schema Object format. Reflects the Open API 3.03 Parameter Object.
      string Key: the name of the parameter. Parameter names are case
      sensitive. Schema Value: the Schema defining the type used for the
      parameter. For function with no parameters, this can be left unset.
      Parameter names must start with a letter or an underscore and must only
      contain chars a-z, A-Z, 0-9, or underscores with a maximum length of 64.
      Example with 1 required and 1 optional parameter: type: OBJECT
      properties: param1: type: STRING param2: type: INTEGER required: -
      param1
    response: Optional. Describes the output from this function in JSON Schema
      format. Reflects the Open API 3.03 Response Object. The Schema defines
      the type used for the response value of the function.
  """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1Schema', 3)
    response = _messages.MessageField('GoogleCloudAiplatformV1beta1Schema', 4)