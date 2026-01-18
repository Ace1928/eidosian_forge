from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
class rv_generic:
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """

    def __init__(self, seed=None):
        super().__init__()
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = sig.varkw is not None or 'moments' in sig.args or 'moments' in sig.kwonlyargs
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """Get or set the generator object for generating random variates.

        If `random_state` is None (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
            self._attach_methods()
        except ValueError:
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_* instance.

        This method must be overridden by subclasses, and must itself call
         _attach_argparser_methods. This method is called in __init__ in
         subclasses, and in __setstate__
        """
        raise NotImplementedError

    def _attach_argparser_methods(self):
        """
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.

        Should be called from `_attach_methods`, typically in __init__ and
        during unpickling (__setstate__)
        """
        ns = {}
        exec(self._parse_arg_template, ns)
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))

    def _construct_argparser(self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser string for the shape arguments.

        This method should be called in __init__ of a class for each
        distribution. It creates the `_parse_arg_template` attribute that is
        then used by `_attach_argparser_methods` to dynamically create and
        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`
        methods to the instance.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """
        if self.shapes:
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')
            shapes = self.shapes.replace(',', ' ').split()
            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError('shapes must be valid python identifiers')
        else:
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)
                args = shapes_args.args[1:]
                if args:
                    shapes_list.append(args)
                    if shapes_args.varargs is not None:
                        raise TypeError('*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError('**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError('kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')
            if shapes_list:
                shapes = shapes_list[0]
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''
        dct = dict(shape_arg_str=shapes_str, locscale_in=locscale_in, locscale_out=locscale_out)
        self._parse_arg_template = parse_arg_template % dct
        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''
        if shapes_vals is None:
            shapes_vals = ()
        vals = ', '.join(('%.3g' % val for val in shapes_vals))
        tempdict['vals'] = vals
        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','
        if self.shapes:
            tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
        else:
            tempdict['set_vals_stmt'] = ''
        if self.shapes is None:
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace('\n%(shapes)s : array_like\n    shape parameters', '')
        for i in range(2):
            if self.shapes is None:
                self.__doc__ = self.__doc__.replace('%(shapes)s, ', '')
            try:
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                raise Exception(f'Unable to construct docstring for distribution "{self.name}": {repr(e)}') from e
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None, docdict=None, discrete='continuous'):
        """Construct instance docstring from the default template."""
        if longname is None:
            longname = 'A'
        self.__doc__ = ''.join([f'{longname} {discrete} random variable.', '\n\n%(before_notes)s\n', docheaders['notes'], '\n%(example)s'])
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    def _stats(self, *args, **kwds):
        return (None, None, None, None)

    def _munp(self, n, *args):
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim
        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,) * -ndiff + bcast_shape
        elif ndiff > 0:
            size_ = (1,) * ndiff + size_
        ok = all([bcdim == 1 or bcdim == szdim for bcdim, szdim in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError(f'size does not match the broadcast shape of the parameters. {size}, {size_}, {bcast_shape}')
        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]
        return (param_bcast, loc_bcast, scale_bcast, size_)

    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, asarray(arg) > 0)
        return cond

    def _get_support(self, *args, **kwargs):
        """Return the support of the (unscaled, unshifted) distribution.

        *Must* be overridden by distributions which have support dependent
        upon the shape parameters of the distribution.  Any such override
        *must not* set or change any of the class members, as these members
        are shared amongst all instances of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support for the specified
            shape parameters.
        """
        return (self.a, self.b)

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0 - self._cdf(x, *args)

    def _logsf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0 - q, *args)

    def rvs(self, *args, **kwds):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `random_state` is None (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance, that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), scale >= 0)
        if not np.all(cond):
            message = f'Domain error in arguments. The `scale` parameter must be positive for all distributions, and many distributions have restrictions on shape parameters. Please see the `scipy.stats.{self.name}` documentation for details.'
            raise ValueError(message)
        if np.all(scale == 0):
            return loc * ones(size, 'd')
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state
        vals = self._rvs(*args, size=size, random_state=random_state)
        vals = vals * scale + loc
        if rndm is not None:
            self._random_state = random_state_saved
        if discrete and (not isinstance(self, rv_sample)):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)
        return vals

    def stats(self, *args, **kwds):
        """Some statistics of the given RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional (continuous RVs only)
            scale parameter (default=1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        stats : sequence
            of requested moments.

        """
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *args + (scale, loc))
            scale, loc, goodargs = (goodargs[-2], goodargs[-1], goodargs[:-2])
            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs, **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)
            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)
            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    with np.errstate(invalid='ignore'):
                        mu2 = np.where(~np.isinf(mu), mu2p - mu ** 2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)
            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)
            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    if g1 is None:
                        mu3 = None
                    else:
                        mu3 = g1 * np.power(mu2, 1.5)
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu ** 2 - 6 * mu2) * mu - 4 * mu3) * mu + mu4p
                        g2 = mu4 / mu2 ** 2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:
            output = [default.copy() for _ in moments]
        output = [out[()] for out in output]
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        """Differential entropy of the RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional  (continuous distributions only).
            Scale parameter (default=1).

        Notes
        -----
        Entropy is defined base `e`:

        >>> import numpy as np
        >>> from scipy.stats._distn_infrastructure import rv_discrete
        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
        >>> np.allclose(drv.entropy(), np.log(2.0))
        True

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, 1 - cond0, self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output[()]

    def moment(self, order, *args, **kwds):
        """non-central moment of distribution of specified order.

        Parameters
        ----------
        order : int, order >= 1
            Order of moment.
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        """
        n = order
        shapes, loc, scale = self._parse_args(*args, **kwds)
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        *shapes, loc, scale = args
        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        i1 = np.logical_and(i0, loc == 0)
        i2 = np.logical_and(i0, loc != 0)
        args = argsreduce(i0, *shapes, loc, scale)
        *shapes, loc, scale = args
        if floor(n) != n:
            raise ValueError('Moment must be an integer.')
        if n < 0:
            raise ValueError('Moment must be positive.')
        mu, mu2, g1, g2 = (None, None, None, None)
        if n > 0 and n < 5:
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*shapes, **mdict)
        val = np.empty(loc.shape)
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)
        result = zeros(i0.shape)
        place(result, ~i0, self.badvalue)
        if i1.any():
            res1 = scale[loc == 0] ** n * val[loc == 0]
            place(result, i1, res1)
        if i2.any():
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            if any(idx):
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            mu, mu2, g1, g2 = mom
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            *shapes, loc, scale, val = args
            res2 = zeros(loc.shape, dtype='d')
            fac = scale / loc
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, shapes)
                res2 += comb(n, k, exact=True) * fac ** k * valk
            res2 += fac ** n * val
            res2 *= loc ** n
            place(result, i2, res2)
        return result[()]

    def median(self, *args, **kwds):
        """Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        rv_discrete.ppf
            Inverse of the CDF

        """
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        mean : float
            the mean of the distribution

        """
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        var : float
            the variance of the distribution

        """
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        std : float
            standard deviation of the distribution

        """
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, confidence, *args, **kwds):
        """Confidence interval with equal areas around the median.

        Parameters
        ----------
        confidence : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        Notes
        -----
        This is implemented as ``ppf([p_tail, 1-p_tail])``, where
        ``ppf`` is the inverse cumulative distribution function and
        ``p_tail = (1-confidence)/2``. Suppose ``[c, d]`` is the support of a
        discrete distribution; then ``ppf([0, 1]) == (c-1, d)``. Therefore,
        when ``confidence=1`` and the distribution is discrete, the left end
        of the interval will be beyond the support of the distribution.
        For discrete distributions, the interval will limit the probability
        in each tail to be less than or equal to ``p_tail`` (usually
        strictly less).

        """
        alpha = confidence
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError('alpha must be between 0 and 1 inclusive')
        q1 = (1.0 - alpha) / 2
        q2 = (1.0 + alpha) / 2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return (a, b)

    def support(self, *args, **kwargs):
        """Support of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : array_like
            end-points of the distribution's support.

        """
        args, loc, scale = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = (arrs[:-2], arrs[-2], arrs[-1])
        cond = self._argcheck(*args) & (scale > 0)
        _a, _b = self._get_support(*args)
        if cond.all():
            return (_a * scale + loc, _b * scale + loc)
        elif cond.ndim == 0:
            return (self.badvalue, self.badvalue)
        _a, _b = (np.asarray(_a).astype('d'), np.asarray(_b).astype('d'))
        out_a, out_b = (_a * scale + loc, _b * scale + loc)
        place(out_a, 1 - cond, self.badvalue)
        place(out_b, 1 - cond, self.badvalue)
        return (out_a, out_b)

    def nnlf(self, theta, x):
        """Negative loglikelihood function.
        Notes
        -----
        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
        parameters (including loc and scale).
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (asarray(x) - loc) / scale
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        return -np.sum(self._logpxf(x, *args), axis=0)

    def _nlff_and_penalty(self, x, args, log_fitfun):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logff = log_fitfun(x, *args)
        finite_logff = np.isfinite(logff)
        n_bad += np.sum(~finite_logff, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logff[finite_logff], axis=0) + penalty
        return -np.sum(logff, axis=0)

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x - loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nlff_and_penalty(x, args, self._logpxf) + n_log_scale

    def _penalized_nlpsf(self, theta, x):
        """Penalized negative log product spacing function.
        i.e., - sum (log (diff (cdf (x, theta))), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        Follows reference [1] of scipy.stats.fit
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (np.sort(x) - loc) / scale

        def log_psf(x, *args):
            x, lj = np.unique(x, return_counts=True)
            cdf_data = self._cdf(x, *args) if x.size else []
            if not (x.size and 1 - cdf_data[-1] <= 0):
                cdf = np.concatenate(([0], cdf_data, [1]))
                lj = np.concatenate((lj, [1]))
            else:
                cdf = np.concatenate(([0], cdf_data))
            return lj * np.log(np.diff(cdf) / lj)
        return self._nlff_and_penalty(x, args, log_psf)