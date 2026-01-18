from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler that writes to a Streamlit app."""

    def __init__(self, parent_container: DeltaGenerator, *, max_thought_containers: int=4, expand_new_thoughts: bool=True, collapse_completed_thoughts: bool=True, thought_labeler: Optional[LLMThoughtLabeler]=None):
        """Create a StreamlitCallbackHandler instance.

        Parameters
        ----------
        parent_container
            The `st.container` that will contain all the Streamlit elements that the
            Handler creates.
        max_thought_containers
            The max number of completed LLM thought containers to show at once. When
            this threshold is reached, a new thought will cause the oldest thoughts to
            be collapsed into a "History" expander. Defaults to 4.
        expand_new_thoughts
            Each LLM "thought" gets its own `st.expander`. This param controls whether
            that expander is expanded by default. Defaults to True.
        collapse_completed_thoughts
            If True, LLM thought expanders will be collapsed when completed.
            Defaults to True.
        thought_labeler
            An optional custom LLMThoughtLabeler instance. If unspecified, the handler
            will use the default thought labeling logic. Defaults to None.
        """
        self._parent_container = parent_container
        self._history_parent = parent_container.container()
        self._history_container: Optional[MutableExpander] = None
        self._current_thought: Optional[LLMThought] = None
        self._completed_thoughts: List[LLMThought] = []
        self._max_thought_containers = max(max_thought_containers, 1)
        self._expand_new_thoughts = expand_new_thoughts
        self._collapse_completed_thoughts = collapse_completed_thoughts
        self._thought_labeler = thought_labeler or LLMThoughtLabeler()

    def _require_current_thought(self) -> LLMThought:
        """Return our current LLMThought. Raise an error if we have no current
        thought.
        """
        if self._current_thought is None:
            raise RuntimeError('Current LLMThought is unexpectedly None!')
        return self._current_thought

    def _get_last_completed_thought(self) -> Optional[LLMThought]:
        """Return our most recent completed LLMThought, or None if we don't have one."""
        if len(self._completed_thoughts) > 0:
            return self._completed_thoughts[len(self._completed_thoughts) - 1]
        return None

    @property
    def _num_thought_containers(self) -> int:
        """The number of 'thought containers' we're currently showing: the
        number of completed thought containers, the history container (if it exists),
        and the current thought container (if it exists).
        """
        count = len(self._completed_thoughts)
        if self._history_container is not None:
            count += 1
        if self._current_thought is not None:
            count += 1
        return count

    def _complete_current_thought(self, final_label: Optional[str]=None) -> None:
        """Complete the current thought, optionally assigning it a new label.
        Add it to our _completed_thoughts list.
        """
        thought = self._require_current_thought()
        thought.complete(final_label)
        self._completed_thoughts.append(thought)
        self._current_thought = None

    def _prune_old_thought_containers(self) -> None:
        """If we have too many thoughts onscreen, move older thoughts to the
        'history container.'
        """
        while self._num_thought_containers > self._max_thought_containers and len(self._completed_thoughts) > 0:
            if self._history_container is None and self._max_thought_containers > 1:
                self._history_container = MutableExpander(self._history_parent, label=self._thought_labeler.get_history_label(), expanded=False)
            oldest_thought = self._completed_thoughts.pop(0)
            if self._history_container is not None:
                self._history_container.markdown(oldest_thought.container.label)
                self._history_container.append_copy(oldest_thought.container)
            oldest_thought.clear()

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThought(parent_container=self._parent_container, expanded=self._expand_new_thoughts, collapse_on_complete=self._collapse_completed_thoughts, labeler=self._thought_labeler)
        self._current_thought.on_llm_start(serialized, prompts)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_new_token(token, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_end(self, output: Any, color: Optional[str]=None, observation_prefix: Optional[str]=None, llm_prefix: Optional[str]=None, **kwargs: Any) -> None:
        output = str(output)
        self._require_current_thought().on_tool_end(output, color, observation_prefix, llm_prefix, **kwargs)
        self._complete_current_thought()

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._require_current_thought().on_tool_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_text(self, text: str, color: Optional[str]=None, end: str='', **kwargs: Any) -> None:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, action: AgentAction, color: Optional[str]=None, **kwargs: Any) -> Any:
        self._require_current_thought().on_agent_action(action, color, **kwargs)
        self._prune_old_thought_containers()

    def on_agent_finish(self, finish: AgentFinish, color: Optional[str]=None, **kwargs: Any) -> None:
        if self._current_thought is not None:
            self._current_thought.complete(self._thought_labeler.get_final_agent_thought_label())
            self._current_thought = None