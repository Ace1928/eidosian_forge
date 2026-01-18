import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
class TestBLEUvsMteval13a(unittest.TestCase):

    def test_corpus_bleu(self):
        ref_file = find('models/wmt15_eval/ref.ru')
        hyp_file = find('models/wmt15_eval/google.ru')
        mteval_output_file = find('models/wmt15_eval/mteval-13a.output')
        with open(mteval_output_file) as mteval_fin:
            mteval_bleu_scores = map(float, mteval_fin.readlines()[-2].split()[1:-1])
        with open(ref_file, encoding='utf8') as ref_fin:
            with open(hyp_file, encoding='utf8') as hyp_fin:
                hypothesis = list(map(lambda x: x.split(), hyp_fin))
                references = list(map(lambda x: [x.split()], ref_fin))
                for i, mteval_bleu in zip(range(1, 10), mteval_bleu_scores):
                    nltk_bleu = corpus_bleu(references, hypothesis, weights=(1.0 / i,) * i)
                    assert abs(mteval_bleu - nltk_bleu) < 0.005
                chencherry = SmoothingFunction()
                for i, mteval_bleu in zip(range(1, 10), mteval_bleu_scores):
                    nltk_bleu = corpus_bleu(references, hypothesis, weights=(1.0 / i,) * i, smoothing_function=chencherry.method3)
                    assert abs(mteval_bleu - nltk_bleu) < 0.005