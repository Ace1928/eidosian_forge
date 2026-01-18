from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def _verified_canonical_retriangulation(M, interval_bits_precs, exact_bits_prec_and_degrees, verbose):
    num_complete_cusps = 0
    num_incomplete_cusps = 0
    for cusp_info in M.cusp_info():
        if cusp_info['complete?']:
            num_complete_cusps += 1
        else:
            num_incomplete_cusps += 1
    if not num_complete_cusps:
        if verbose:
            print('Failure: Due to no unfilled cusp.')
            print('Next step: Give up.')
        return None
    if num_incomplete_cusps:
        Mfilled = M.filled_triangulation()
    else:
        Mfilled = M
    Mcopy = _retrying_high_precision_canonize(Mfilled)
    if not Mcopy:
        if verbose:
            print("Failure: In SnapPea kernel's proto_canonize()")
            print('Next step: Give up.')
        return None
    if interval_bits_precs:
        for interval_bits_prec in interval_bits_precs:
            if verbose:
                print('Method: Intervals with interval_bits_prec = %d' % interval_bits_prec)
            try:
                return interval_checked_canonical_triangulation(Mcopy, interval_bits_prec)
            except (RuntimeError, exceptions.NumericalVerifyError) as e:
                if verbose:
                    _print_exception(e)
                    if isinstance(e, exceptions.NumericalVerifyError):
                        print('Failure: Could not verify proto-canonical triangulation.')
                    else:
                        print('Failure: Could not find verified interval.')
                    print('Next step: trying different method/precision.')
    if exact_bits_prec_and_degrees:
        for bits_prec, degree in exact_bits_prec_and_degrees:
            if verbose:
                print('Method: Exact, using LLL with bits_prec = %d, degree = %d' % (bits_prec, degree))
            try:
                return exactly_checked_canonical_retriangulation(Mcopy, bits_prec, degree)
            except FindExactShapesError as e:
                if verbose:
                    _print_exception(e)
                    print('Failure: Could not find exact shapes.')
                    print('Next step: trying different method/precision')
    return None