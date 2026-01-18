from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
class AffineScalarFunc(object):
    """
    Affine functions that support basic mathematical operations
    (addition, etc.).  Such functions can for instance be used for
    representing the local (linear) behavior of any function.

    This class can also be used to represent constants.

    The variables of affine scalar functions are Variable objects.

    AffineScalarFunc objects include facilities for calculating the
    'error' on the function, from the uncertainties on its variables.

    Main attributes and methods:

    - nominal_value, std_dev: value at the origin / nominal value, and
      standard deviation.  The standard deviation can be NaN or infinity.

    - n, s: abbreviations for nominal_value and std_dev.

    - error_components(): error_components()[x] is the error due to
      Variable x.

    - derivatives: derivatives[x] is the (value of the) derivative
      with respect to Variable x.  This attribute is a Derivatives
      dictionary whose keys are the Variable objects on which the
      function depends. The values are the numerical values of the
      derivatives.

      All the Variable objects on which the function depends are in
      'derivatives'.

    - std_score(x): position of number x with respect to the
      nominal value, in units of the standard deviation.
    """
    __slots__ = ('_nominal_value', '_linear_part')

    class dtype(object):
        type = staticmethod(lambda value: value)

    def __init__(self, nominal_value, linear_part):
        """
        nominal_value -- value of the function when the linear part is
        zero.

        linear_part -- LinearCombination that describes the linear
        part of the AffineScalarFunc.
        """
        self._nominal_value = float(nominal_value)
        self._linear_part = linear_part

    @property
    def nominal_value(self):
        """Nominal value of the random number."""
        return self._nominal_value
    n = nominal_value

    @property
    def derivatives(self):
        """
        Return a mapping from each Variable object on which the function
        (self) depends to the value of the derivative with respect to
        that variable.

        This mapping should not be modified.

        Derivative values are always floats.

        This mapping is cached, for subsequent calls.
        """
        if not self._linear_part.expanded():
            self._linear_part.expand()
            self._linear_part.linear_combo.default_factory = None
        return self._linear_part.linear_combo

    def __bool__(self):
        """
        Equivalent to self != 0.
        """
        return self != 0.0
    __eq__ = force_aff_func_args(eq_on_aff_funcs)
    __ne__ = force_aff_func_args(ne_on_aff_funcs)
    __gt__ = force_aff_func_args(gt_on_aff_funcs)
    __ge__ = force_aff_func_args(ge_on_aff_funcs)
    __lt__ = force_aff_func_args(lt_on_aff_funcs)
    __le__ = force_aff_func_args(le_on_aff_funcs)

    def error_components(self):
        """
        Individual components of the standard deviation of the affine
        function (in absolute value), returned as a dictionary with
        Variable objects as keys. The returned variables are the
        independent variables that the affine function depends on.

        This method assumes that the derivatives contained in the
        object take scalar values (and are not a tuple, like what
        math.frexp() returns, for instance).
        """
        error_components = {}
        for variable, derivative in self.derivatives.items():
            if variable._std_dev == 0:
                error_components[variable] = 0
            else:
                error_components[variable] = abs(derivative * variable._std_dev)
        return error_components

    @property
    def std_dev(self):
        """
        Standard deviation of the affine function.

        This method assumes that the function returns scalar results.

        This returned standard deviation depends on the current
        standard deviations [std_dev] of the variables (Variable
        objects) involved.
        """
        return CallableStdDev(sqrt(sum((delta ** 2 for delta in self.error_components().values()))))
    s = std_dev

    def __repr__(self):
        std_dev = self.std_dev
        if std_dev:
            std_dev_str = repr(std_dev)
        else:
            std_dev_str = '0'
        return '%r+/-%s' % (self.nominal_value, std_dev_str)

    def __str__(self):
        return self.format('')

    def __format__(self, format_spec):
        """
        Formats a number with uncertainty.

        The format specification are the same as for format() for
        floats, as defined for Python 2.6+ (restricted to what the %
        operator accepts, if using an earlier version of Python),
        except that the n presentation type is not supported. In
        particular, the usual precision, alignment, sign flag,
        etc. can be used. The behavior of the various presentation
        types (e, f, g, none, etc.) is similar. Moreover, the format
        is extended: the number of digits of the uncertainty can be
        controlled, as is the way the uncertainty is indicated (with
        +/- or with the short-hand notation 3.14(1), in LaTeX or with
        a simple text string,...).

        Beyond the use of options at the end of the format
        specification, the main difference with floats is that a "u"
        just before the presentation type (f, e, g, none, etc.)
        activates the "uncertainty control" mode (e.g.: ".6u").  This
        mode is also activated when not using any explicit precision
        (e.g.: "g", "10f", "+010,e" format specifications).  If the
        uncertainty does not have a meaningful number of significant
        digits (0 and NaN uncertainties), this mode is automatically
        deactivated.

        The nominal value and the uncertainty always use the same
        precision. This implies trailing zeros, in general, even with
        the g format type (contrary to the float case). However, when
        the number of significant digits of the uncertainty is not
        defined (zero or NaN uncertainty), it has no precision, so
        there is no matching. In this case, the original format
        specification is used for the nominal value (any "u" is
        ignored).

        Any precision (".p", where p is a number) is interpreted (if
        meaningful), in the uncertainty control mode, as indicating
        the number p of significant digits of the displayed
        uncertainty. Example: .1uf will return a string with one
        significant digit in the uncertainty (and no exponent).

        If no precision is given, the rounding rules from the
        Particle Data Group are used, if possible
        (http://pdg.lbl.gov/2010/reviews/rpp2010-rev-rpp-intro.pdf). For
        example, the "f" format specification generally does not use
        the default 6 digits after the decimal point, but applies the
        PDG rules.

        A common exponent is used if an exponent is needed for the
        larger of the nominal value (in absolute value) and the
        standard deviation, unless this would result in a zero
        uncertainty being represented as 0e... or a NaN uncertainty as
        NaNe.... Thanks to this common exponent, the quantity that
        best describes the associated probability distribution has a
        mantissa in the usual 1-10 range. The common exponent is
        factored (as in "(1.2+/-0.1)e-5"). unless the format
        specification contains an explicit width (" 1.2e-5+/- 0.1e-5")
        (this allows numbers to be in a single column, when printing
        numbers over many lines). Specifying a minimum width of 1 is a
        way of forcing any common exponent to not be factored out.

        The fill, align, zero and width parameters of the format
        specification are applied individually to each of the nominal
        value and standard deviation or, if the shorthand notation is
        used, globally.

        The sign parameter of the format specification is only applied
        to the nominal value (since the standard deviation is
        positive).

        In the case of a non-LaTeX output, the returned string can
        normally be parsed back with ufloat_fromstr(). This however
        excludes cases where numbers use the "," thousands separator,
        for example.

        Options can be added, at the end of the format
        specification. Multiple options can be specified:

        - When "P" is present, the pretty-printing mode is activated: "±"
          separates the nominal value from the standard deviation, exponents
          use superscript characters, etc.
        - When "S" is present (like in .1uS), the short-hand notation 1.234(5)
          is used, indicating an uncertainty on the last digits; if the digits
          of the uncertainty straddle the decimal point, it uses a fixed-point
          notation, like in 12.3(4.5).
        - When "L" is present, the output is formatted with LaTeX.
        - "p" ensures that there are parentheses around the …±… part (no
          parentheses are added if some are already present, for instance
          because of an exponent or of a trailing % sign, etc.). This produces
          outputs like (1.0±0.2) or (1.0±0.2)e7, which can be useful for
          removing any ambiguity if physical units are added after the printed
          number.
    
        An uncertainty which is exactly zero is represented as the
        integer 0 (i.e. with no decimal point).

        The "%" format type forces the percent sign to be at the end
        of the returned string (it is not attached to each of the
        nominal value and the standard deviation).

        Some details of the formatting can be customized as described
        in format_num().
        """
        match = re.match('\n            (?P<fill>[^{}]??)(?P<align>[<>=^]?)  # fill cannot be { or }\n            (?P<sign>[-+ ]?)\n            (?P<zero>0?)\n            (?P<width>\\d*)\n            (?P<comma>,?)\n            (?:\\.(?P<prec>\\d+))?\n            (?P<uncert_prec>u?)  # Precision for the uncertainty?\n            # The type can be omitted. Options must not go here:\n            (?P<type>[eEfFgG%]??)  # n not supported\n            (?P<options>[PSLp]*)  # uncertainties-specific flags\n            $', format_spec, re.VERBOSE)
        if not match:
            raise ValueError('Format specification %r cannot be used with object of type %r. Note that uncertainties-specific flags must be put at the end of the format string.' % (format_spec, self.__class__.__name__))
        pres_type = match.group('type') or None
        fmt_prec = match.group('prec')
        nom_val = self.nominal_value
        std_dev = self.std_dev
        options = set(match.group('options'))
        if pres_type == '%':
            std_dev *= 100
            nom_val *= 100
            pres_type = 'f'
            options.add('%')
        real_values = [value for value in [abs(nom_val), std_dev] if not isinfinite(value)]
        if pres_type in (None, 'e', 'E', 'g', 'G'):
            try:
                exp_ref_value = max(real_values)
            except ValueError:
                pass
        if (not fmt_prec and len(real_values) == 2 or match.group('uncert_prec')) and std_dev and (not isinfinite(std_dev)):
            if fmt_prec:
                num_signif_d = int(fmt_prec)
                if not num_signif_d:
                    raise ValueError('The number of significant digits on the uncertainty should be positive')
            else:
                num_signif_d, std_dev = PDG_precision(std_dev)
            digits_limit = signif_dgt_to_limit(std_dev, num_signif_d)
        else:
            if fmt_prec:
                prec = int(fmt_prec)
            elif pres_type is None:
                prec = 12
            else:
                prec = 6
            if pres_type in ('f', 'F'):
                digits_limit = -prec
            else:
                if pres_type in ('e', 'E'):
                    num_signif_digits = prec + 1
                else:
                    num_signif_digits = prec or 1
                digits_limit = signif_dgt_to_limit(exp_ref_value, num_signif_digits) if real_values else None
        if pres_type in ('f', 'F'):
            use_exp = False
        elif pres_type in ('e', 'E'):
            if not real_values:
                use_exp = False
            else:
                use_exp = True
                common_exp = first_digit(round(exp_ref_value, -digits_limit))
        elif not real_values:
            use_exp = False
        else:
            common_exp = first_digit(round(exp_ref_value, -digits_limit))
            if -4 <= common_exp < common_exp - digits_limit + 1:
                use_exp = False
            else:
                use_exp = True
        if use_exp:
            factor = 10.0 ** common_exp
            nom_val_mantissa = nom_val / factor
            std_dev_mantissa = std_dev / factor
            signif_limit = digits_limit - common_exp
        else:
            common_exp = None
            nom_val_mantissa = nom_val
            std_dev_mantissa = std_dev
            signif_limit = digits_limit
        main_pres_type = 'fF'[(pres_type or 'g').isupper()]
        if signif_limit is not None:
            prec = max(-signif_limit, 1 if pres_type is None and (not std_dev) else 0)
        return format_num(nom_val_mantissa, std_dev_mantissa, common_exp, match.groupdict(), prec=prec, main_pres_type=main_pres_type, options=options)

    @set_doc("\n        Return the same result as self.__format__(format_spec), or\n        equivalently as the format(self, format_spec) of Python 2.6+.\n\n        This method is meant to be used for formatting numbers with\n        uncertainties in Python < 2.6, with '... %s ...' %\n        num.format('.2e').\n        ")
    def format(*args, **kwargs):
        return args[0].__format__(*args[1:], **kwargs)

    def std_score(self, value):
        """
        Return 'value' - nominal value, in units of the standard
        deviation.

        Raises a ValueError exception if the standard deviation is zero.
        """
        try:
            return (value - self._nominal_value) / self.std_dev
        except ZeroDivisionError:
            raise ValueError('The standard deviation is zero: undefined result')

    def __deepcopy__(self, memo):
        """
        Hook for the standard copy module.

        The returned AffineScalarFunc is a completely fresh copy,
        which is fully independent of any variable defined so far.
        New variables are specially created for the returned
        AffineScalarFunc object.
        """
        return AffineScalarFunc(self._nominal_value, copy.deepcopy(self._linear_part))

    def __getstate__(self):
        """
        Hook for the pickle module.

        The slot attributes of the parent classes are returned, as
        well as those of the __dict__ attribute of the object (if
        any).
        """
        all_attrs = {}
        try:
            all_attrs['__dict__'] = self.__dict__
        except AttributeError:
            pass
        all_slots = set()
        for cls in type(self).mro():
            slot_names = getattr(cls, '__slots__', ())
            if isinstance(slot_names, basestring):
                all_slots.add(slot_names)
            else:
                all_slots.update(slot_names)
        for name in all_slots:
            try:
                all_attrs[name] = getattr(self, name)
            except AttributeError:
                pass
        return all_attrs

    def __setstate__(self, data_dict):
        """
        Hook for the pickle module.
        """
        for name, value in data_dict.items():
            setattr(self, name, value)