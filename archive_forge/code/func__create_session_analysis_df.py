from __future__ import annotations
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def _create_session_analysis_df(self) -> Any:
    """Create a dataframe with all the information from the session."""
    pd = import_pandas()
    on_llm_end_records_df = pd.DataFrame(self.on_llm_end_records)
    llm_input_prompts_df = ClearMLCallbackHandler._build_llm_df(base_df=on_llm_end_records_df, base_df_fields=['step', 'prompts'] + (['name'] if 'name' in on_llm_end_records_df else ['id']), rename_map={'step': 'prompt_step'})
    complexity_metrics_columns = []
    visualizations_columns: List = []
    if self.complexity_metrics:
        complexity_metrics_columns = ['flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index', 'coleman_liau_index', 'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula', 'gunning_fog', 'text_standard', 'fernandez_huerta', 'szigriszt_pazos', 'gutierrez_polini', 'crawford', 'gulpease_index', 'osman']
    llm_outputs_df = ClearMLCallbackHandler._build_llm_df(on_llm_end_records_df, ['step', 'text', 'token_usage_total_tokens', 'token_usage_prompt_tokens', 'token_usage_completion_tokens'] + complexity_metrics_columns + visualizations_columns, {'step': 'output_step', 'text': 'output'})
    session_analysis_df = pd.concat([llm_input_prompts_df, llm_outputs_df], axis=1)
    return session_analysis_df