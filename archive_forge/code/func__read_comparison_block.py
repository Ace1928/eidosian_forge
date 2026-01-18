import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
def _read_comparison_block(self, stream):
    while True:
        line = stream.readline()
        if not line:
            return []
        comparison_tags = re.findall(COMPARISON, line)
        if comparison_tags:
            grad_comparisons = re.findall(GRAD_COMPARISON, line)
            non_grad_comparisons = re.findall(NON_GRAD_COMPARISON, line)
            comparison_text = stream.readline().strip()
            if self._word_tokenizer:
                comparison_text = self._word_tokenizer.tokenize(comparison_text)
            stream.readline()
            comparison_bundle = []
            if grad_comparisons:
                for comp in grad_comparisons:
                    comp_type = int(re.match('<cs-(\\d)>', comp).group(1))
                    comparison = Comparison(text=comparison_text, comp_type=comp_type)
                    line = stream.readline()
                    entities_feats = ENTITIES_FEATS.findall(line)
                    if entities_feats:
                        for code, entity_feat in entities_feats:
                            if code == '1':
                                comparison.entity_1 = entity_feat.strip()
                            elif code == '2':
                                comparison.entity_2 = entity_feat.strip()
                            elif code == '3':
                                comparison.feature = entity_feat.strip()
                    keyword = KEYWORD.findall(line)
                    if keyword:
                        comparison.keyword = keyword[0]
                    comparison_bundle.append(comparison)
            if non_grad_comparisons:
                for comp in non_grad_comparisons:
                    comp_type = int(re.match('<cs-(\\d)>', comp).group(1))
                    comparison = Comparison(text=comparison_text, comp_type=comp_type)
                    comparison_bundle.append(comparison)
            return comparison_bundle