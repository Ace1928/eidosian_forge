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
class OpenAiAgent(Agent):
    """
    Agent that uses the openai API to generate code.

    <Tip warning={true}>

    The openAI models are used in generation mode, so even for the `chat()` API, it's better to use models like
    `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

    </Tip>

    Args:
        model (`str`, *optional*, defaults to `"text-davinci-003"`):
            The name of the OpenAI model to use.
        api_key (`str`, *optional*):
            The API key to use. If unset, will look for the environment variable `"OPENAI_API_KEY"`.
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
    from transformers import OpenAiAgent

    agent = OpenAiAgent(model="text-davinci-003", api_key=xxx)
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(self, model='text-davinci-003', api_key=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        if not is_openai_available():
            raise ImportError('Using `OpenAiAgent` requires `openai`: `pip install openai`.')
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise ValueError("You need an openai key to use `OpenAIAgent`. You can get one here: Get one here https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = xxx.")
        else:
            openai.api_key = api_key
        self.model = model
        super().__init__(chat_prompt_template=chat_prompt_template, run_prompt_template=run_prompt_template, additional_tools=additional_tools)

    def generate_many(self, prompts, stop):
        if 'gpt' in self.model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        else:
            return self._completion_generate(prompts, stop)

    def generate_one(self, prompt, stop):
        if 'gpt' in self.model:
            return self._chat_generate(prompt, stop)
        else:
            return self._completion_generate([prompt], stop)[0]

    def _chat_generate(self, prompt, stop):
        result = openai.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': prompt}], temperature=0, stop=stop)
        return result.choices[0].message.content

    def _completion_generate(self, prompts, stop):
        result = openai.Completion.create(model=self.model, prompt=prompts, temperature=0, stop=stop, max_tokens=200)
        return [answer['text'] for answer in result['choices']]