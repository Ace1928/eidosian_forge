import io
import json
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Mapping, Optional, TypeVar, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
class SagemakerEndpoint(LLM):
    """Sagemaker Inference Endpoint models.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Sagemaker endpoint.
    See: https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html
    """
    '\n    Args:        \n\n        region_name: The aws region e.g., `us-west-2`.\n            Fallsback to AWS_DEFAULT_REGION env variable\n            or region specified in ~/.aws/config.\n\n        credentials_profile_name: The name of the profile in the ~/.aws/credentials\n            or ~/.aws/config files, which has either access keys or role information\n            specified. If not specified, the default credential profile or, if on an\n            EC2 instance, credentials from IMDS will be used.\n\n        client: boto3 client for Sagemaker Endpoint\n\n        content_handler: Implementation for model specific LLMContentHandler \n\n\n    Example:\n        .. code-block:: python\n\n            from langchain_community.llms import SagemakerEndpoint\n            endpoint_name = (\n                "my-endpoint-name"\n            )\n            region_name = (\n                "us-west-2"\n            )\n            credentials_profile_name = (\n                "default"\n            )\n            se = SagemakerEndpoint(\n                endpoint_name=endpoint_name,\n                region_name=region_name,\n                credentials_profile_name=credentials_profile_name\n            )\n\n        #Use with boto3 client\n            client = boto3.client(\n                        "sagemaker-runtime",\n                        region_name=region_name\n                    )\n\n            se = SagemakerEndpoint(\n                endpoint_name=endpoint_name,\n                client=client\n            )\n\n    '
    client: Any = None
    'Boto3 client for sagemaker runtime'
    endpoint_name: str = ''
    'The name of the endpoint from the deployed Sagemaker model.\n    Must be unique within an AWS Region.'
    region_name: str = ''
    'The aws region where the Sagemaker model is deployed, eg. `us-west-2`.'
    credentials_profile_name: Optional[str] = None
    'The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which\n    has either access keys or role information specified.\n    If not specified, the default credential profile or, if on an EC2 instance,\n    credentials from IMDS will be used.\n    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html\n    '
    content_handler: LLMContentHandler
    'The content handler class that provides an input and\n    output transform functions to handle formats between LLM\n    and the endpoint.\n    '
    streaming: bool = False
    'Whether to stream the results.'
    '\n     Example:\n        .. code-block:: python\n\n        from langchain_community.llms.sagemaker_endpoint import LLMContentHandler\n\n        class ContentHandler(LLMContentHandler):\n                content_type = "application/json"\n                accepts = "application/json"\n\n                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:\n                    input_str = json.dumps({prompt: prompt, **model_kwargs})\n                    return input_str.encode(\'utf-8\')\n                \n                def transform_output(self, output: bytes) -> str:\n                    response_json = json.loads(output.read().decode("utf-8"))\n                    return response_json[0]["generated_text"]\n    '
    model_kwargs: Optional[Dict] = None
    'Keyword arguments to pass to the model.'
    endpoint_kwargs: Optional[Dict] = None
    'Optional attributes passed to the invoke_endpoint\n    function. See `boto3`_. docs for more info.\n    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>\n    '

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Dont do anything if client provided externally"""
        if values.get('client') is not None:
            return values
        'Validate that AWS credentials to and python package exists in environment.'
        try:
            import boto3
            try:
                if values['credentials_profile_name'] is not None:
                    session = boto3.Session(profile_name=values['credentials_profile_name'])
                else:
                    session = boto3.Session()
                values['client'] = session.client('sagemaker-runtime', region_name=values['region_name'])
            except Exception as e:
                raise ValueError('Could not load credentials to authenticate with AWS client. Please check that credentials in the specified profile name are valid.') from e
        except ImportError:
            raise ImportError('Could not import boto3 python package. Please install it with `pip install boto3`.')
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {**{'endpoint_name': self.endpoint_name}, **{'model_kwargs': _model_kwargs}}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'sagemaker_endpoint'

    def _call(self, prompt: str, stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> str:
        """Call out to Sagemaker inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        body = self.content_handler.transform_input(prompt, _model_kwargs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts
        if self.streaming and run_manager:
            try:
                resp = self.client.invoke_endpoint_with_response_stream(EndpointName=self.endpoint_name, Body=body, ContentType=self.content_handler.content_type, **_endpoint_kwargs)
                iterator = LineIterator(resp['Body'])
                current_completion: str = ''
                for line in iterator:
                    resp = json.loads(line)
                    resp_output = resp.get('outputs')[0]
                    if stop is not None:
                        resp_output = enforce_stop_tokens(resp_output, stop)
                    current_completion += resp_output
                    run_manager.on_llm_new_token(resp_output)
                return current_completion
            except Exception as e:
                raise ValueError(f'Error raised by streaming inference endpoint: {e}')
        else:
            try:
                response = self.client.invoke_endpoint(EndpointName=self.endpoint_name, Body=body, ContentType=content_type, Accept=accepts, **_endpoint_kwargs)
            except Exception as e:
                raise ValueError(f'Error raised by inference endpoint: {e}')
            text = self.content_handler.transform_output(response['Body'])
            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            return text