import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
class AzureOpenAiAgent(Agent):
    """
    Agent that uses Azure OpenAI to generate code. See the [official
    documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) to learn how to deploy an openAI
    model on Azure

    <Tip warning={true}>

    The openAI models are used in generation mode, so even for the `chat()` API, it's better to use models like
    `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

    </Tip>

    Args:
        deployment_id (`str`):
            The name of the deployed Azure openAI model to use.
        api_key (`str`, *optional*):
            The API key to use. If unset, will look for the environment variable `"AZURE_OPENAI_API_KEY"`.
        resource_name (`str`, *optional*):
            The name of your Azure OpenAI Resource. If unset, will look for the environment variable
            `"AZURE_OPENAI_RESOURCE_NAME"`.
        api_version (`str`, *optional*, default to `"2022-12-01"`):
            The API version to use for this agent.
        is_chat_mode (`bool`, *optional*):
            Whether you are using a completion model or a chat model (see note above, chat models won't be as
            efficient). Will default to `gpt` being in the `deployment_id` or not.
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import AzureOpenAiAgent

    agent = AzureAiAgent(deployment_id="Davinci-003", api_key=xxx, resource_name=yyy)
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(self, deployment_id, api_key=None, resource_name=None, api_version='2022-12-01', is_chat_model=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        if not is_openai_available():
            raise ImportError('Using `OpenAiAgent` requires `openai`: `pip install openai`.')
        self.deployment_id = deployment_id
        openai.api_type = 'azure'
        if api_key is None:
            api_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
        if api_key is None:
            raise ValueError("You need an Azure openAI key to use `AzureOpenAIAgent`. If you have one, set it in your env with `os.environ['AZURE_OPENAI_API_KEY'] = xxx.")
        else:
            openai.api_key = api_key
        if resource_name is None:
            resource_name = os.environ.get('AZURE_OPENAI_RESOURCE_NAME', None)
        if resource_name is None:
            raise ValueError("You need a resource_name to use `AzureOpenAIAgent`. If you have one, set it in your env with `os.environ['AZURE_OPENAI_RESOURCE_NAME'] = xxx.")
        else:
            openai.api_base = f'https://{resource_name}.openai.azure.com'
        openai.api_version = api_version
        if is_chat_model is None:
            is_chat_model = 'gpt' in deployment_id.lower()
        self.is_chat_model = is_chat_model
        super().__init__(chat_prompt_template=chat_prompt_template, run_prompt_template=run_prompt_template, additional_tools=additional_tools)

    def generate_many(self, prompts, stop):
        if self.is_chat_model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        else:
            return self._completion_generate(prompts, stop)

    def generate_one(self, prompt, stop):
        if self.is_chat_model:
            return self._chat_generate(prompt, stop)
        else:
            return self._completion_generate([prompt], stop)[0]

    def _chat_generate(self, prompt, stop):
        result = openai.ChatCompletion.create(engine=self.deployment_id, messages=[{'role': 'user', 'content': prompt}], temperature=0, stop=stop)
        return result['choices'][0]['message']['content']

    def _completion_generate(self, prompts, stop):
        result = openai.Completion.create(engine=self.deployment_id, prompt=prompts, temperature=0, stop=stop, max_tokens=200)
        return [answer['text'] for answer in result['choices']]