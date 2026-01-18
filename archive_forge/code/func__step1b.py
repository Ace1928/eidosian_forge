import re
from nltk.stem.api import StemmerI
def _step1b(self, word):
    """Implements Step 1b from "An algorithm for suffix stripping"

        From the paper:

            (m>0) EED -> EE                    feed      ->  feed
                                               agreed    ->  agree
            (*v*) ED  ->                       plastered ->  plaster
                                               bled      ->  bled
            (*v*) ING ->                       motoring  ->  motor
                                               sing      ->  sing

        If the second or third of the rules in Step 1b is successful,
        the following is done:

            AT -> ATE                       conflat(ed)  ->  conflate
            BL -> BLE                       troubl(ed)   ->  trouble
            IZ -> IZE                       siz(ed)      ->  size
            (*d and not (*L or *S or *Z))
               -> single letter
                                            hopp(ing)    ->  hop
                                            tann(ed)     ->  tan
                                            fall(ing)    ->  fall
                                            hiss(ing)    ->  hiss
                                            fizz(ed)     ->  fizz
            (m=1 and *o) -> E               fail(ing)    ->  fail
                                            fil(ing)     ->  file

        The rule to map to a single letter causes the removal of one of
        the double letter pair. The -E is put back on -AT, -BL and -IZ,
        so that the suffixes -ATE, -BLE and -IZE can be recognised
        later. This E may be removed in step 4.
        """
    if self.mode == self.NLTK_EXTENSIONS:
        if word.endswith('ied'):
            if len(word) == 4:
                return self._replace_suffix(word, 'ied', 'ie')
            else:
                return self._replace_suffix(word, 'ied', 'i')
    if word.endswith('eed'):
        stem = self._replace_suffix(word, 'eed', '')
        if self._measure(stem) > 0:
            return stem + 'ee'
        else:
            return word
    rule_2_or_3_succeeded = False
    for suffix in ['ed', 'ing']:
        if word.endswith(suffix):
            intermediate_stem = self._replace_suffix(word, suffix, '')
            if self._contains_vowel(intermediate_stem):
                rule_2_or_3_succeeded = True
                break
    if not rule_2_or_3_succeeded:
        return word
    return self._apply_rule_list(intermediate_stem, [('at', 'ate', None), ('bl', 'ble', None), ('iz', 'ize', None), ('*d', intermediate_stem[-1], lambda stem: intermediate_stem[-1] not in ('l', 's', 'z')), ('', 'e', lambda stem: self._measure(stem) == 1 and self._ends_cvc(stem))])