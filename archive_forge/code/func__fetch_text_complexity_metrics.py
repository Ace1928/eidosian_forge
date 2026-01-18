import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def _fetch_text_complexity_metrics(text: str) -> dict:
    textstat = import_textstat()
    text_complexity_metrics = {'flesch_reading_ease': textstat.flesch_reading_ease(text), 'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text), 'smog_index': textstat.smog_index(text), 'coleman_liau_index': textstat.coleman_liau_index(text), 'automated_readability_index': textstat.automated_readability_index(text), 'dale_chall_readability_score': textstat.dale_chall_readability_score(text), 'difficult_words': textstat.difficult_words(text), 'linsear_write_formula': textstat.linsear_write_formula(text), 'gunning_fog': textstat.gunning_fog(text), 'text_standard': textstat.text_standard(text), 'fernandez_huerta': textstat.fernandez_huerta(text), 'szigriszt_pazos': textstat.szigriszt_pazos(text), 'gutierrez_polini': textstat.gutierrez_polini(text), 'crawford': textstat.crawford(text), 'gulpease_index': textstat.gulpease_index(text), 'osman': textstat.osman(text)}
    return text_complexity_metrics