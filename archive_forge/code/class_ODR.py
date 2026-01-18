import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
class ODR:
    """
    The ODR class gathers all information and coordinates the running of the
    main fitting routine.

    Members of instances of the ODR class have the same names as the arguments
    to the initialization routine.

    Parameters
    ----------
    data : Data class instance
        instance of the Data class
    model : Model class instance
        instance of the Model class

    Other Parameters
    ----------------
    beta0 : array_like of rank-1
        a rank-1 sequence of initial parameter values. Optional if
        model provides an "estimate" function to estimate these values.
    delta0 : array_like of floats of rank-1, optional
        a (double-precision) float array to hold the initial values of
        the errors in the input variables. Must be same shape as data.x
    ifixb : array_like of ints of rank-1, optional
        sequence of integers with the same length as beta0 that determines
        which parameters are held fixed. A value of 0 fixes the parameter,
        a value > 0 makes the parameter free.
    ifixx : array_like of ints with same shape as data.x, optional
        an array of integers with the same shape as data.x that determines
        which input observations are treated as fixed. One can use a sequence
        of length m (the dimensionality of the input observations) to fix some
        dimensions for all observations. A value of 0 fixes the observation,
        a value > 0 makes it free.
    job : int, optional
        an integer telling ODRPACK what tasks to perform. See p. 31 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_job post-initialization for a more readable interface.
    iprint : int, optional
        an integer telling ODRPACK what to print. See pp. 33-34 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_iprint post-initialization for a more readable interface.
    errfile : str, optional
        string with the filename to print ODRPACK errors to. If the file already
        exists, an error will be thrown. The `overwrite` argument can be used to
        prevent this. *Do Not Open This File Yourself!*
    rptfile : str, optional
        string with the filename to print ODRPACK summaries to. If the file
        already exists, an error will be thrown. The `overwrite` argument can be
        used to prevent this. *Do Not Open This File Yourself!*
    ndigit : int, optional
        integer specifying the number of reliable digits in the computation
        of the function.
    taufac : float, optional
        float specifying the initial trust region. The default value is 1.
        The initial trust region is equal to taufac times the length of the
        first computed Gauss-Newton step. taufac must be less than 1.
    sstol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the sum-of-squares. The default value is eps**(1/2) where eps
        is the smallest value such that 1 + eps > 1 for double precision
        computation on the machine. sstol must be less than 1.
    partol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the estimated parameters. The default value is eps**(2/3) for
        explicit models and ``eps**(1/3)`` for implicit models. partol must be less
        than 1.
    maxit : int, optional
        integer specifying the maximum number of iterations to perform. For
        first runs, maxit is the total number of iterations performed and
        defaults to 50. For restarts, maxit is the number of additional
        iterations to perform and defaults to 10.
    stpb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
        finite difference derivatives wrt the parameters.
    stpd : optional
        array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
        step sizes to compute finite difference derivatives wrt the input
        variable errors. If stpd is a rank-1 array with length m (the
        dimensionality of the input variable), then the values are broadcast to
        all observations.
    sclb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of scaling factors for the
        parameters. The purpose of these scaling factors are to scale all of
        the parameters to around unity. Normally appropriate scaling factors
        are computed if this argument is not specified. Specify them yourself
        if the automatic procedure goes awry.
    scld : array_like, optional
        array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
        factors for the *errors* in the input variables. Again, these factors
        are automatically computed if you do not provide them. If scld.shape ==
        (m,), then the scaling factors are broadcast to all observations.
    work : ndarray, optional
        array to hold the double-valued working data for ODRPACK. When
        restarting, takes the value of self.output.work.
    iwork : ndarray, optional
        array to hold the integer-valued working data for ODRPACK. When
        restarting, takes the value of self.output.iwork.
    overwrite : bool, optional
        If it is True, output files defined by `errfile` and `rptfile` are
        overwritten. The default is False.

    Attributes
    ----------
    data : Data
        The data for this fit
    model : Model
        The model used in fit
    output : Output
        An instance if the Output class containing all of the returned
        data from an invocation of ODR.run() or ODR.restart()

    """

    def __init__(self, data, model, beta0=None, delta0=None, ifixb=None, ifixx=None, job=None, iprint=None, errfile=None, rptfile=None, ndigit=None, taufac=None, sstol=None, partol=None, maxit=None, stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None, overwrite=False):
        self.data = data
        self.model = model
        if beta0 is None:
            if self.model.estimate is not None:
                self.beta0 = _conv(self.model.estimate(self.data))
            else:
                raise ValueError('must specify beta0 or provide an estimator with the model')
        else:
            self.beta0 = _conv(beta0)
        if ifixx is None and data.fix is not None:
            ifixx = data.fix
        if overwrite:
            if rptfile is not None and os.path.exists(rptfile):
                os.remove(rptfile)
            if errfile is not None and os.path.exists(errfile):
                os.remove(errfile)
        self.delta0 = _conv(delta0)
        self.ifixx = _conv(ifixx, dtype=numpy.int32)
        self.ifixb = _conv(ifixb, dtype=numpy.int32)
        self.job = job
        self.iprint = iprint
        self.errfile = errfile
        self.rptfile = rptfile
        self.ndigit = ndigit
        self.taufac = taufac
        self.sstol = sstol
        self.partol = partol
        self.maxit = maxit
        self.stpb = _conv(stpb)
        self.stpd = _conv(stpd)
        self.sclb = _conv(sclb)
        self.scld = _conv(scld)
        self.work = _conv(work)
        self.iwork = _conv(iwork)
        self.output = None
        self._check()

    def _check(self):
        """ Check the inputs for consistency, but don't bother checking things
        that the builtin function odr will check.
        """
        x_s = list(self.data.x.shape)
        if isinstance(self.data.y, numpy.ndarray):
            y_s = list(self.data.y.shape)
            if self.model.implicit:
                raise OdrError('an implicit model cannot use response data')
        else:
            y_s = [self.data.y, x_s[-1]]
            if not self.model.implicit:
                raise OdrError('an explicit model needs response data')
            self.set_job(fit_type=1)
        if x_s[-1] != y_s[-1]:
            raise OdrError('number of observations do not match')
        n = x_s[-1]
        if len(x_s) == 2:
            m = x_s[0]
        else:
            m = 1
        if len(y_s) == 2:
            q = y_s[0]
        else:
            q = 1
        p = len(self.beta0)
        fcn_perms = [(q, n)]
        fjacd_perms = [(q, m, n)]
        fjacb_perms = [(q, p, n)]
        if q == 1:
            fcn_perms.append((n,))
            fjacd_perms.append((m, n))
            fjacb_perms.append((p, n))
        if m == 1:
            fjacd_perms.append((q, n))
        if p == 1:
            fjacb_perms.append((q, n))
        if m == q == 1:
            fjacd_perms.append((n,))
        if p == q == 1:
            fjacb_perms.append((n,))
        arglist = (self.beta0, self.data.x)
        if self.model.extra_args is not None:
            arglist = arglist + self.model.extra_args
        res = self.model.fcn(*arglist)
        if res.shape not in fcn_perms:
            print(res.shape)
            print(fcn_perms)
            raise OdrError('fcn does not output %s-shaped array' % y_s)
        if self.model.fjacd is not None:
            res = self.model.fjacd(*arglist)
            if res.shape not in fjacd_perms:
                raise OdrError('fjacd does not output %s-shaped array' % repr((q, m, n)))
        if self.model.fjacb is not None:
            res = self.model.fjacb(*arglist)
            if res.shape not in fjacb_perms:
                raise OdrError('fjacb does not output %s-shaped array' % repr((q, p, n)))
        if self.delta0 is not None and self.delta0.shape != self.data.x.shape:
            raise OdrError('delta0 is not a %s-shaped array' % repr(self.data.x.shape))
        if self.data.x.size == 0:
            warn('Empty data detected for ODR instance. Do not expect any fitting to occur', OdrWarning, stacklevel=3)

    def _gen_work(self):
        """ Generate a suitable work array if one does not already exist.
        """
        n = self.data.x.shape[-1]
        p = self.beta0.shape[0]
        if len(self.data.x.shape) == 2:
            m = self.data.x.shape[0]
        else:
            m = 1
        if self.model.implicit:
            q = self.data.y
        elif len(self.data.y.shape) == 2:
            q = self.data.y.shape[0]
        else:
            q = 1
        if self.data.we is None:
            ldwe = ld2we = 1
        elif len(self.data.we.shape) == 3:
            ld2we, ldwe = self.data.we.shape[1:]
        else:
            we = self.data.we
            ldwe = 1
            ld2we = 1
            if we.ndim == 1 and q == 1:
                ldwe = n
            elif we.ndim == 2:
                if we.shape == (q, q):
                    ld2we = q
                elif we.shape == (q, n):
                    ldwe = n
        if self.job % 10 < 2:
            lwork = 18 + 11 * p + p * p + m + m * m + 4 * n * q + 6 * n * m + 2 * n * q * p + 2 * n * q * m + q * q + 5 * q + q * (p + m) + ldwe * ld2we * q
        else:
            lwork = 18 + 11 * p + p * p + m + m * m + 4 * n * q + 2 * n * m + 2 * n * q * p + 5 * q + q * (p + m) + ldwe * ld2we * q
        if isinstance(self.work, numpy.ndarray) and self.work.shape == (lwork,) and self.work.dtype.str.endswith('f8'):
            return
        else:
            self.work = numpy.zeros((lwork,), float)

    def set_job(self, fit_type=None, deriv=None, var_calc=None, del_init=None, restart=None):
        """
        Sets the "job" parameter is a hopefully comprehensible way.

        If an argument is not specified, then the value is left as is. The
        default value from class initialization is for all of these options set
        to 0.

        Parameters
        ----------
        fit_type : {0, 1, 2} int
            0 -> explicit ODR

            1 -> implicit ODR

            2 -> ordinary least-squares
        deriv : {0, 1, 2, 3} int
            0 -> forward finite differences

            1 -> central finite differences

            2 -> user-supplied derivatives (Jacobians) with results
              checked by ODRPACK

            3 -> user-supplied derivatives, no checking
        var_calc : {0, 1, 2} int
            0 -> calculate asymptotic covariance matrix and fit
                 parameter uncertainties (V_B, s_B) using derivatives
                 recomputed at the final solution

            1 -> calculate V_B and s_B using derivatives from last iteration

            2 -> do not calculate V_B and s_B
        del_init : {0, 1} int
            0 -> initial input variable offsets set to 0

            1 -> initial offsets provided by user in variable "work"
        restart : {0, 1} int
            0 -> fit is not a restart

            1 -> fit is a restart

        Notes
        -----
        The permissible values are different from those given on pg. 31 of the
        ODRPACK User's Guide only in that one cannot specify numbers greater than
        the last value for each variable.

        If one does not supply functions to compute the Jacobians, the fitting
        procedure will change deriv to 0, finite differences, as a default. To
        initialize the input variable offsets by yourself, set del_init to 1 and
        put the offsets into the "work" variable correctly.

        """
        if self.job is None:
            job_l = [0, 0, 0, 0, 0]
        else:
            job_l = [self.job // 10000 % 10, self.job // 1000 % 10, self.job // 100 % 10, self.job // 10 % 10, self.job % 10]
        if fit_type in (0, 1, 2):
            job_l[4] = fit_type
        if deriv in (0, 1, 2, 3):
            job_l[3] = deriv
        if var_calc in (0, 1, 2):
            job_l[2] = var_calc
        if del_init in (0, 1):
            job_l[1] = del_init
        if restart in (0, 1):
            job_l[0] = restart
        self.job = job_l[0] * 10000 + job_l[1] * 1000 + job_l[2] * 100 + job_l[3] * 10 + job_l[4]

    def set_iprint(self, init=None, so_init=None, iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
        """ Set the iprint parameter for the printing of computation reports.

        If any of the arguments are specified here, then they are set in the
        iprint member. If iprint is not set manually or with this method, then
        ODRPACK defaults to no printing. If no filename is specified with the
        member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
        print to stdout in addition to the specified filename by setting the
        so_* arguments to this function, but one cannot specify to print to
        stdout but not a file since one can do that by not specifying a rptfile
        filename.

        There are three reports: initialization, iteration, and final reports.
        They are represented by the arguments init, iter, and final
        respectively.  The permissible values are 0, 1, and 2 representing "no
        report", "short report", and "long report" respectively.

        The argument iter_step (0 <= iter_step <= 9) specifies how often to make
        the iteration report; the report will be made for every iter_step'th
        iteration starting with iteration one. If iter_step == 0, then no
        iteration report is made, regardless of the other arguments.

        If the rptfile is None, then any so_* arguments supplied will raise an
        exception.
        """
        if self.iprint is None:
            self.iprint = 0
        ip = [self.iprint // 1000 % 10, self.iprint // 100 % 10, self.iprint // 10 % 10, self.iprint % 10]
        ip2arg = [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]
        if self.rptfile is None and (so_init is not None or so_iter is not None or so_final is not None):
            raise OdrError('no rptfile specified, cannot output to stdout twice')
        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]
        if init is not None:
            iprint_l[0] = init
        if so_init is not None:
            iprint_l[1] = so_init
        if iter is not None:
            iprint_l[2] = iter
        if so_iter is not None:
            iprint_l[3] = so_iter
        if final is not None:
            iprint_l[4] = final
        if so_final is not None:
            iprint_l[5] = so_final
        if iter_step in range(10):
            ip[2] = iter_step
        ip[0] = ip2arg.index(iprint_l[0:2])
        ip[1] = ip2arg.index(iprint_l[2:4])
        ip[3] = ip2arg.index(iprint_l[4:6])
        self.iprint = ip[0] * 1000 + ip[1] * 100 + ip[2] * 10 + ip[3]

    def run(self):
        """ Run the fitting routine with all of the information given and with ``full_output=1``.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """
        args = (self.model.fcn, self.beta0, self.data.y, self.data.x)
        kwds = {'full_output': 1}
        kwd_l = ['ifixx', 'ifixb', 'job', 'iprint', 'errfile', 'rptfile', 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb', 'stpd', 'sclb', 'scld', 'work', 'iwork']
        if self.delta0 is not None and self.job // 10000 % 10 == 0:
            self._gen_work()
            d0 = numpy.ravel(self.delta0)
            self.work[:len(d0)] = d0
        if self.model.fjacb is not None:
            kwds['fjacb'] = self.model.fjacb
        if self.model.fjacd is not None:
            kwds['fjacd'] = self.model.fjacd
        if self.data.we is not None:
            kwds['we'] = self.data.we
        if self.data.wd is not None:
            kwds['wd'] = self.data.wd
        if self.model.extra_args is not None:
            kwds['extra_args'] = self.model.extra_args
        for attr in kwd_l:
            obj = getattr(self, attr)
            if obj is not None:
                kwds[attr] = obj
        self.output = Output(odr(*args, **kwds))
        return self.output

    def restart(self, iter=None):
        """ Restarts the run with iter more iterations.

        Parameters
        ----------
        iter : int, optional
            ODRPACK's default for the number of new iterations is 10.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """
        if self.output is None:
            raise OdrError('cannot restart: run() has not been called before')
        self.set_job(restart=1)
        self.work = self.output.work
        self.iwork = self.output.iwork
        self.maxit = iter
        return self.run()