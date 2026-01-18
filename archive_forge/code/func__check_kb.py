from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir
def _check_kb(kb):
    assert kb.get_size_entities() == 4
    for entity_string in ['Q53', 'Q17', 'Q007', 'Q44']:
        assert entity_string in kb.get_entity_strings()
    for entity_string in ['', 'Q0']:
        assert entity_string not in kb.get_entity_strings()
    assert kb.get_size_aliases() == 3
    for alias_string in ['double07', 'guy', 'random']:
        assert alias_string in kb.get_alias_strings()
    for alias_string in ['nothingness', '', 'randomnoise']:
        assert alias_string not in kb.get_alias_strings()
    candidates = sorted(kb.get_alias_candidates('double07'), key=lambda x: x.entity_)
    assert len(candidates) == 2
    assert candidates[0].entity_ == 'Q007'
    assert 6.999 < candidates[0].entity_freq < 7.01
    assert candidates[0].entity_vector == [0, 0, 7]
    assert candidates[0].alias_ == 'double07'
    assert 0.899 < candidates[0].prior_prob < 0.901
    assert candidates[1].entity_ == 'Q17'
    assert 1.99 < candidates[1].entity_freq < 2.01
    assert candidates[1].entity_vector == [7, 1, 0]
    assert candidates[1].alias_ == 'double07'
    assert 0.099 < candidates[1].prior_prob < 0.101