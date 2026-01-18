from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
class ContractExpression:
    """Helper class for storing an explicit ``contraction_list`` which can
    then be repeatedly called solely with the array arguments.
    """

    def __init__(self, contraction, contraction_list, constants_dict, **einsum_kwargs):
        self.contraction_list = contraction_list
        self.einsum_kwargs = einsum_kwargs
        self.contraction = format_const_einsum_str(contraction, constants_dict.keys())
        self._full_num_args = contraction.count(',') + 1
        self.num_args = self._full_num_args - len(constants_dict)
        self._full_contraction_list = contraction_list
        self._constants_dict = constants_dict
        self._evaluated_constants = {}
        self._backend_expressions = {}

    def evaluate_constants(self, backend='auto'):
        """Convert any constant operands to the correct backend form, and
        perform as many contractions as possible to create a new list of
        operands, stored in ``self._evaluated_constants[backend]``. This also
        makes sure ``self.contraction_list`` only contains the remaining,
        non-const operations.
        """
        tmp_const_ops = [self._constants_dict.get(i, None) for i in range(self._full_num_args)]
        backend = parse_backend(tmp_const_ops, backend)
        try:
            new_ops, new_contraction_list = backends.evaluate_constants(backend, tmp_const_ops, self)
        except KeyError:
            new_ops, new_contraction_list = self(*tmp_const_ops, backend=backend, evaluate_constants=True)
        self._evaluated_constants[backend] = new_ops
        self.contraction_list = new_contraction_list

    def _get_evaluated_constants(self, backend):
        """Retrieve or generate the cached list of constant operators (mixed
        in with None representing non-consts) and the remaining contraction
        list.
        """
        try:
            return self._evaluated_constants[backend]
        except KeyError:
            self.evaluate_constants(backend)
            return self._evaluated_constants[backend]

    def _get_backend_expression(self, arrays, backend):
        try:
            return self._backend_expressions[backend]
        except KeyError:
            fn = backends.build_expression(backend, arrays, self)
            self._backend_expressions[backend] = fn
            return fn

    def _contract(self, arrays, out=None, backend='auto', evaluate_constants=False):
        """The normal, core contraction.
        """
        contraction_list = self._full_contraction_list if evaluate_constants else self.contraction_list
        return _core_contract(list(arrays), contraction_list, out=out, backend=backend, evaluate_constants=evaluate_constants, **self.einsum_kwargs)

    def _contract_with_conversion(self, arrays, out, backend, evaluate_constants=False):
        """Special contraction, i.e., contraction with a different backend
        but converting to and from that backend. Retrieves or generates a
        cached expression using ``arrays`` as templates, then calls it
        with ``arrays``.

        If ``evaluate_constants=True``, perform a partial contraction that
        prepares the constant tensors and operations with the right backend.
        """
        if evaluate_constants:
            return backends.evaluate_constants(backend, arrays, self)
        result = self._get_backend_expression(arrays, backend)(*arrays)
        if out is not None:
            out[()] = result
            return out
        return result

    def __call__(self, *arrays, **kwargs):
        """Evaluate this expression with a set of arrays.

        Parameters
        ----------
        arrays : seq of array
            The arrays to supply as input to the expression.
        out : array, optional (default: ``None``)
            If specified, output the result into this array.
        backend : str, optional  (default: ``numpy``)
            Perform the contraction with this backend library. If numpy arrays
            are supplied then try to convert them to and from the correct
            backend array type.
        """
        out = kwargs.pop('out', None)
        backend = kwargs.pop('backend', 'auto')
        backend = parse_backend(arrays, backend)
        evaluate_constants = kwargs.pop('evaluate_constants', False)
        if kwargs:
            raise ValueError('The only valid keyword arguments to a `ContractExpression` call are `out=` or `backend=`. Got: {}.'.format(kwargs))
        correct_num_args = self._full_num_args if evaluate_constants else self.num_args
        if len(arrays) != correct_num_args:
            raise ValueError('This `ContractExpression` takes exactly {} array arguments but received {}.'.format(self.num_args, len(arrays)))
        if self._constants_dict and (not evaluate_constants):
            ops_var, ops_const = (iter(arrays), self._get_evaluated_constants(backend))
            ops = [next(ops_var) if op is None else op for op in ops_const]
        else:
            ops = arrays
        try:
            if backends.has_backend(backend) and all((isinstance(x, np.ndarray) for x in arrays)):
                return self._contract_with_conversion(ops, out, backend, evaluate_constants=evaluate_constants)
            return self._contract(ops, out, backend, evaluate_constants=evaluate_constants)
        except ValueError as err:
            original_msg = str(err.args) if err.args else ''
            msg = ("Internal error while evaluating `ContractExpression`. Note that few checks are performed - the number and rank of the array arguments must match the original expression. The internal error was: '{}'".format(original_msg),)
            err.args = msg
            raise

    def __repr__(self):
        if self._constants_dict:
            constants_repr = ', constants={}'.format(sorted(self._constants_dict))
        else:
            constants_repr = ''
        return "<ContractExpression('{}'{})>".format(self.contraction, constants_repr)

    def __str__(self):
        s = [self.__repr__()]
        for i, c in enumerate(self.contraction_list):
            s.append('\n  {}.  '.format(i + 1))
            s.append("'{}'".format(c[2]) + (' [{}]'.format(c[-1]) if c[-1] else ''))
        if self.einsum_kwargs:
            s.append('\neinsum_kwargs={}'.format(self.einsum_kwargs))
        return ''.join(s)