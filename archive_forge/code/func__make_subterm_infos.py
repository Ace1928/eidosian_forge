import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def _make_subterm_infos(terms, num_column_counts, cat_levels_contrasts):
    term_buckets = OrderedDict()
    bucket_ordering = []
    for term in terms:
        num_factors = []
        for factor in term.factors:
            if factor in num_column_counts:
                num_factors.append(factor)
        bucket = frozenset(num_factors)
        if bucket not in term_buckets:
            bucket_ordering.append(bucket)
        term_buckets.setdefault(bucket, []).append(term)
    if frozenset() in term_buckets:
        bucket_ordering.remove(frozenset())
        bucket_ordering.insert(0, frozenset())
    term_to_subterm_infos = OrderedDict()
    new_term_order = []
    for bucket in bucket_ordering:
        bucket_terms = term_buckets[bucket]
        bucket_terms.sort(key=lambda t: len(t.factors))
        new_term_order += bucket_terms
        used_subterms = set()
        for term in bucket_terms:
            subterm_infos = []
            factor_codings = pick_contrasts_for_term(term, num_column_counts, used_subterms)
            for factor_coding in factor_codings:
                subterm_factors = []
                contrast_matrices = {}
                subterm_columns = 1
                for factor in term.factors:
                    if factor in num_column_counts:
                        subterm_factors.append(factor)
                        subterm_columns *= num_column_counts[factor]
                    elif factor in factor_coding:
                        subterm_factors.append(factor)
                        levels, contrast = cat_levels_contrasts[factor]
                        coded = code_contrast_matrix(factor_coding[factor], levels, contrast, default=Treatment)
                        contrast_matrices[factor] = coded
                        subterm_columns *= coded.matrix.shape[1]
                subterm_infos.append(SubtermInfo(subterm_factors, contrast_matrices, subterm_columns))
            term_to_subterm_infos[term] = subterm_infos
    assert new_term_order == list(term_to_subterm_infos)
    return term_to_subterm_infos