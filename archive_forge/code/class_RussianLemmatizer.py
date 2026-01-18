from typing import Callable, Dict, List, Optional, Tuple
from thinc.api import Model
from ...pipeline import Lemmatizer
from ...pipeline.lemmatizer import lemmatizer_score
from ...symbols import POS
from ...tokens import Token
from ...vocab import Vocab
class RussianLemmatizer(Lemmatizer):

    def __init__(self, vocab: Vocab, model: Optional[Model], name: str='lemmatizer', *, mode: str='pymorphy3', overwrite: bool=False, scorer: Optional[Callable]=lemmatizer_score) -> None:
        if mode in {'pymorphy2', 'pymorphy2_lookup'}:
            try:
                from pymorphy2 import MorphAnalyzer
            except ImportError:
                raise ImportError("The lemmatizer mode 'pymorphy2' requires the pymorphy2 library and dictionaries. Install them with: pip install pymorphy2# for Ukrainian dictionaries:pip install pymorphy2-dicts-uk") from None
            if getattr(self, '_morph', None) is None:
                self._morph = MorphAnalyzer(lang='ru')
        elif mode in {'pymorphy3', 'pymorphy3_lookup'}:
            try:
                from pymorphy3 import MorphAnalyzer
            except ImportError:
                raise ImportError("The lemmatizer mode 'pymorphy3' requires the pymorphy3 library and dictionaries. Install them with: pip install pymorphy3# for Ukrainian dictionaries:pip install pymorphy3-dicts-uk") from None
            if getattr(self, '_morph', None) is None:
                self._morph = MorphAnalyzer(lang='ru')
        super().__init__(vocab, model, name, mode=mode, overwrite=overwrite, scorer=scorer)

    def _pymorphy_lemmatize(self, token: Token) -> List[str]:
        string = token.text
        univ_pos = token.pos_
        morphology = token.morph.to_dict()
        if univ_pos == 'PUNCT':
            return [PUNCT_RULES.get(string, string)]
        if univ_pos not in ('ADJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB'):
            return self._pymorphy_lookup_lemmatize(token)
        analyses = self._morph.parse(string)
        filtered_analyses = []
        for analysis in analyses:
            if not analysis.is_known:
                continue
            analysis_pos, _ = oc2ud(str(analysis.tag))
            if analysis_pos == univ_pos or (analysis_pos in ('NOUN', 'PROPN') and univ_pos in ('NOUN', 'PROPN')) or (analysis_pos == 'PRON' and univ_pos == 'DET'):
                filtered_analyses.append(analysis)
        if not len(filtered_analyses):
            return [string.lower()]
        if morphology is None or (len(morphology) == 1 and POS in morphology):
            return list(dict.fromkeys([analysis.normal_form for analysis in filtered_analyses]))
        if univ_pos in ('ADJ', 'DET', 'NOUN', 'PROPN'):
            features_to_compare = ['Case', 'Number', 'Gender']
        elif univ_pos == 'NUM':
            features_to_compare = ['Case', 'Gender']
        elif univ_pos == 'PRON':
            features_to_compare = ['Case', 'Number', 'Gender', 'Person']
        else:
            features_to_compare = ['Aspect', 'Gender', 'Mood', 'Number', 'Tense', 'VerbForm', 'Voice']
        analyses, filtered_analyses = (filtered_analyses, [])
        for analysis in analyses:
            _, analysis_morph = oc2ud(str(analysis.tag))
            for feature in features_to_compare:
                if feature in morphology and feature in analysis_morph and (morphology[feature].lower() != analysis_morph[feature].lower()):
                    break
            else:
                filtered_analyses.append(analysis)
        if not len(filtered_analyses):
            return [string.lower()]
        return list(dict.fromkeys([analysis.normal_form for analysis in filtered_analyses]))

    def _pymorphy_lookup_lemmatize(self, token: Token) -> List[str]:
        string = token.text
        analyses = self._morph.parse(string)
        normal_forms = set([an.normal_form for an in analyses])
        if len(normal_forms) == 1:
            return [next(iter(normal_forms))]
        return [string]

    def pymorphy2_lemmatize(self, token: Token) -> List[str]:
        return self._pymorphy_lemmatize(token)

    def pymorphy2_lookup_lemmatize(self, token: Token) -> List[str]:
        return self._pymorphy_lookup_lemmatize(token)

    def pymorphy3_lemmatize(self, token: Token) -> List[str]:
        return self._pymorphy_lemmatize(token)

    def pymorphy3_lookup_lemmatize(self, token: Token) -> List[str]:
        return self._pymorphy_lookup_lemmatize(token)