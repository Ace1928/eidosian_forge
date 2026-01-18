from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
class OCIGenAIBase(BaseModel, ABC):
    """Base class for OCI GenAI models"""
    client: Any
    auth_type: Optional[str] = 'API_KEY'
    'Authentication type, could be \n    \n    API_KEY, \n    SECURITY_TOKEN, \n    INSTANCE_PRINCIPLE, \n    RESOURCE_PRINCIPLE\n\n    If not specified, API_KEY will be used\n    '
    auth_profile: Optional[str] = 'DEFAULT'
    'The name of the profile in ~/.oci/config\n    If not specified , DEFAULT will be used \n    '
    model_id: str = None
    'Id of the model to call, e.g., cohere.command'
    provider: str = None
    'Provider name of the model. Default to None, \n    will try to be derived from the model_id\n    otherwise, requires user input\n    '
    model_kwargs: Optional[Dict] = None
    'Keyword arguments to pass to the model'
    service_endpoint: str = None
    'service endpoint url'
    compartment_id: str = None
    'OCID of compartment'
    is_stream: bool = False
    'Whether to stream back partial progress'
    llm_stop_sequence_mapping: Mapping[str, str] = {'cohere': 'stop_sequences', 'meta': 'stop'}

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that OCI config and python package exists in environment."""
        if values['client'] is not None:
            return values
        try:
            import oci
            client_kwargs = {'config': {}, 'signer': None, 'service_endpoint': values['service_endpoint'], 'retry_strategy': oci.retry.DEFAULT_RETRY_STRATEGY, 'timeout': (10, 240)}
            if values['auth_type'] == OCIAuthType(1).name:
                client_kwargs['config'] = oci.config.from_file(profile_name=values['auth_profile'])
                client_kwargs.pop('signer', None)
            elif values['auth_type'] == OCIAuthType(2).name:

                def make_security_token_signer(oci_config):
                    pk = oci.signer.load_private_key_from_file(oci_config.get('key_file'), None)
                    with open(oci_config.get('security_token_file'), encoding='utf-8') as f:
                        st_string = f.read()
                    return oci.auth.signers.SecurityTokenSigner(st_string, pk)
                client_kwargs['config'] = oci.config.from_file(profile_name=values['auth_profile'])
                client_kwargs['signer'] = make_security_token_signer(oci_config=client_kwargs['config'])
            elif values['auth_type'] == OCIAuthType(3).name:
                client_kwargs['signer'] = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            elif values['auth_type'] == OCIAuthType(4).name:
                client_kwargs['signer'] = oci.auth.signers.get_resource_principals_signer()
            else:
                raise ValueError('Please provide valid value to auth_type')
            values['client'] = oci.generative_ai_inference.GenerativeAiInferenceClient(**client_kwargs)
        except ImportError as ex:
            raise ModuleNotFoundError('Could not import oci python package. Please make sure you have the oci package installed.') from ex
        except Exception as e:
            raise ValueError('Could not authenticate with OCI client. Please check if ~/.oci/config exists. If INSTANCE_PRINCIPLE or RESOURCE_PRINCIPLE is used, Please check the specified auth_profile and auth_type are valid.') from e
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {**{'model_kwargs': _model_kwargs}}

    def _get_provider(self) -> str:
        if self.provider is not None:
            provider = self.provider
        else:
            provider = self.model_id.split('.')[0].lower()
        if provider not in VALID_PROVIDERS:
            raise ValueError(f'Invalid provider derived from model_id: {self.model_id} Please explicitly pass in the supported provider when using custom endpoint')
        return provider