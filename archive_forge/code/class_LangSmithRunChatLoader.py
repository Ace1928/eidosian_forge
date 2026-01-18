from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union, cast
from langchain_core.chat_sessions import ChatSession
from langchain_core.load.load import load
from langchain_community.chat_loaders.base import BaseChatLoader
class LangSmithRunChatLoader(BaseChatLoader):
    """
    Load chat sessions from a list of LangSmith "llm" runs.

    Attributes:
        runs (Iterable[Union[str, Run]]): The list of LLM run IDs or run objects.
        client (Client): Instance of LangSmith client for fetching data.
    """

    def __init__(self, runs: Iterable[Union[str, Run]], client: Optional['Client']=None):
        """
        Initialize a new LangSmithRunChatLoader instance.

        :param runs: List of LLM run IDs or run objects.
        :param client: An instance of LangSmith client, if not provided,
            a new client instance will be created.
        """
        from langsmith.client import Client
        self.runs = runs
        self.client = client or Client()

    def _load_single_chat_session(self, llm_run: 'Run') -> ChatSession:
        """
        Convert an individual LangSmith LLM run to a ChatSession.

        :param llm_run: The LLM run object.
        :return: A chat session representing the run's data.
        """
        chat_session = LangSmithRunChatLoader._get_messages_from_llm_run(llm_run)
        functions = LangSmithRunChatLoader._get_functions_from_llm_run(llm_run)
        if functions:
            chat_session['functions'] = functions
        return chat_session

    @staticmethod
    def _get_messages_from_llm_run(llm_run: 'Run') -> ChatSession:
        """
        Extract messages from a LangSmith LLM run.

        :param llm_run: The LLM run object.
        :return: ChatSession with the extracted messages.
        """
        if llm_run.run_type != 'llm':
            raise ValueError(f'Expected run of type llm. Got: {llm_run.run_type}')
        if 'messages' not in llm_run.inputs:
            raise ValueError(f"Run has no 'messages' inputs. Got {llm_run.inputs}")
        if not llm_run.outputs:
            raise ValueError('Cannot convert pending run')
        messages = load(llm_run.inputs)['messages']
        message_chunk = load(llm_run.outputs)['generations'][0]['message']
        return ChatSession(messages=messages + [message_chunk])

    @staticmethod
    def _get_functions_from_llm_run(llm_run: 'Run') -> Optional[List[Dict]]:
        """
        Extract functions from a LangSmith LLM run if they exist.

        :param llm_run: The LLM run object.
        :return: Functions from the run or None.
        """
        if llm_run.run_type != 'llm':
            raise ValueError(f'Expected run of type llm. Got: {llm_run.run_type}')
        return (llm_run.extra or {}).get('invocation_params', {}).get('functions')

    def lazy_load(self) -> Iterator[ChatSession]:
        """
        Lazy load the chat sessions from the iterable of run IDs.

        This method fetches the runs and converts them to chat sessions on-the-fly,
        yielding one session at a time.

        :return: Iterator of chat sessions containing messages.
        """
        from langsmith.schemas import Run
        for run_obj in self.runs:
            try:
                if hasattr(run_obj, 'id'):
                    run = run_obj
                else:
                    run = self.client.read_run(run_obj)
                session = self._load_single_chat_session(cast(Run, run))
                yield session
            except ValueError as e:
                logger.warning(f'Could not load run {run_obj}: {repr(e)}')
                continue