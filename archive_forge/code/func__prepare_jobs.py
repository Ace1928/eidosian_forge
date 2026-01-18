from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _prepare_jobs(ids, nums_frequency, spectra, shifts, atol):
    """For inputs to reconstruct, determine how the given information yields
    function reconstruction tasks and collect them into a dictionary ``jobs``.
    Also determine whether the function at zero is needed.

    Args:
        ids (dict or Sequence or str): Indices for the QNode parameters with respect to which
            the QNode should be reconstructed as a univariate function, per QNode argument.
            Each key of the dict, entry of the list, or the single ``str`` has to be the name
            of an argument of ``qnode`` .
            If a ``dict`` , the values of ``ids`` have to contain the parameter indices
            for the respective array-valued QNode argument represented by the key.
            These indices always are tuples, i.e. ``()`` for scalar and ``(i,)`` for
            one-dimensional arguments.
            If a ``list`` , the parameter indices are inferred from ``nums_frequency`` if
            given or ``spectra`` else.
            If ``None``, all keys present in ``nums_frequency`` / ``spectra`` are considered.
        nums_frequency (dict[dict]): Numbers of integer frequencies -- and biggest
            frequency -- per QNode parameter. The keys have to be argument names of ``qnode``
            and the inner dictionaries have to be mappings from parameter indices to the
            respective integer number of frequencies. If the QNode frequencies are not contiguous
            integers, the argument ``spectra`` should be used to save evaluations of ``qnode`` .
            Takes precedence over ``spectra`` and leads to usage of equidistant shifts.
        spectra (dict[dict]): Frequency spectra per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective frequency spectrum for that
            parameter. Ignored if ``nums_frequency!=None``.
        shifts (dict[dict]): Shift angles for the reconstruction per QNode parameter.
            The keys have to be argument names of ``qnode`` and the inner dictionaries have to
            be mappings from parameter indices to the respective shift angles to be used for that
            parameter. For :math:`R` non-zero frequencies, there must be :math:`2R+1` shifts
            given. Ignored if ``nums_frequency!=None``.
        atol (float): Absolute tolerance used to analyze shifts lying close to 0.

    Returns:
        dict[dict]: Indices for the QNode parameters with respect to which the QNode
            will be reconstructed. Cast to the dictionary structure explained above.
            If the input ``ids`` was a dictionary, it is returned unmodified.
        callable: The reconstruction method to use, one out of two internal methods.
        dict[dict[dict]]: Keyword arguments for the reconstruction method specifying
            how to carry out the reconstruction. The outer-most keys are QNode argument
            names, the middle keys are parameter indices like the inner keys of
            ``nums_frequency`` or ``spectra`` and the inner-most dictionary contains the
            keyword arguments, i.e. the keys are keyword argument names for the
            reconstruction method
        bool: Whether any of the reconstruction jobs will require the evaluation
            of the function at the position of reconstruction itself.
    """
    if nums_frequency is None:
        if spectra is None:
            raise ValueError('Either nums_frequency or spectra must be given.')
        ids = _parse_ids(ids, spectra)
        if shifts is None:
            shifts = {}
        need_f0 = False
        recon_fn = _reconstruct_gen
        jobs = {}
        for arg_name, inner_dict in ids.items():
            _jobs = {}
            for par_idx in inner_dict:
                _spectrum = spectra[arg_name][par_idx]
                R = len(_spectrum) - 1
                _shifts, need_f0 = _parse_shifts(shifts, R, arg_name, par_idx, atol, need_f0)
                if R > 0:
                    _jobs[par_idx] = {'shifts': _shifts, 'spectrum': _spectrum}
                else:
                    _jobs[par_idx] = None
            jobs[arg_name] = _jobs
    else:
        jobs = {}
        need_f0 = True
        ids = _parse_ids(ids, nums_frequency)
        recon_fn = _reconstruct_equ
        for arg_name, inner_dict in ids.items():
            _jobs = {}
            for par_idx in inner_dict:
                _num_frequency = nums_frequency[arg_name][par_idx]
                _jobs[par_idx] = {'num_frequency': _num_frequency} if _num_frequency > 0 else None
            jobs[arg_name] = _jobs
    return (ids, recon_fn, jobs, need_f0)