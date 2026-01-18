import json
import warnings
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.azureml_endpoint import (
class CustomOpenAIChatContentFormatter(ContentFormatterBase):
    """Chat Content formatter for models with OpenAI like API scheme."""
    SUPPORTED_ROLES: List[str] = ['user', 'assistant', 'system']

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> Dict:
        """Converts a message to a dict according to a role"""
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            return {'role': 'user', 'content': ContentFormatterBase.escape_special_characters(content)}
        elif isinstance(message, AIMessage):
            return {'role': 'assistant', 'content': ContentFormatterBase.escape_special_characters(content)}
        elif isinstance(message, SystemMessage):
            return {'role': 'system', 'content': ContentFormatterBase.escape_special_characters(content)}
        elif isinstance(message, ChatMessage) and message.role in CustomOpenAIChatContentFormatter.SUPPORTED_ROLES:
            return {'role': message.role, 'content': ContentFormatterBase.escape_special_characters(content)}
        else:
            supported = ','.join([role for role in CustomOpenAIChatContentFormatter.SUPPORTED_ROLES])
            raise ValueError(f'Received unsupported role. \n                Supported roles for the LLaMa Foundation Model: {supported}')

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.serverless]

    def format_messages_request_payload(self, messages: List[BaseMessage], model_kwargs: Dict, api_type: AzureMLEndpointApiType) -> bytes:
        """Formats the request according to the chosen api"""
        chat_messages = [CustomOpenAIChatContentFormatter._convert_message_to_dict(message) for message in messages]
        if api_type in [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.realtime]:
            request_payload = json.dumps({'input_data': {'input_string': chat_messages, 'parameters': model_kwargs}})
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({'messages': chat_messages, **model_kwargs})
        else:
            raise ValueError(f'`api_type` {api_type} is not supported by this formatter')
        return str.encode(request_payload)

    def format_response_payload(self, output: bytes, api_type: AzureMLEndpointApiType=AzureMLEndpointApiType.dedicated) -> ChatGeneration:
        """Formats response"""
        if api_type in [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.realtime]:
            try:
                choice = json.loads(output)['output']
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(message=BaseMessage(content=choice.strip(), type='assistant'), generation_info=None)
        if api_type == AzureMLEndpointApiType.serverless:
            try:
                choice = json.loads(output)['choices'][0]
                if not isinstance(choice, dict):
                    raise TypeError('Endpoint response is not well formed for a chat model. Expected `dict` but `{type(choice)}` was received.')
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(message=BaseMessage(content=choice['message']['content'].strip(), type=choice['message']['role']), generation_info=dict(finish_reason=choice.get('finish_reason'), logprobs=choice.get('logprobs')))
        raise ValueError(f'`api_type` {api_type} is not supported by this formatter')