import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore
@magic_arguments()
@argument('-i', '--input', action='append', help=textwrap.dedent("\n        Names of Python objects to be assigned to R\n        objects after using the default converter or\n        one specified through the argument `-c/--converter`.\n        Multiple inputs can be passed separated only by commas with no\n        whitespace.\n\n        Names of Python objects can be either the name of an object\n        in the current notebook/ipython shell, or a path to a name\n        nested in a namespace visible from the current notebook/ipython\n        shell. For example, '-i myvariable' or\n        '-i mypackage.myothervariable' would both work.\n\n        Each input can be either the name of Python object, in which\n        case the same name will be used for the R object, or an\n        expression of the form <r-name>=<python-name>."))
@argument('-o', '--output', action='append', help=textwrap.dedent("\n        Names of variables to be pushed from rpy2 to `shell.user_ns` after\n        executing cell body (rpy2's internal facilities will apply ri2ro as\n        appropriate). Multiple names can be passed separated only by commas\n        with no whitespace."))
@argument('-n', '--noreturn', help='Force the magic to not return anything.', action='store_true', default=False)
@argument_group('Plot', 'Arguments to plotting device')
@argument('-w', '--width', type=float, help='Width of plotting device in R.')
@argument('-h', '--height', type=float, help='Height of plotting device in R.')
@argument('-p', '--pointsize', type=int, help='Pointsize of plotting device in R.')
@argument('-b', '--bg', help='Background of plotting device in R.')
@argument_group('SVG', 'SVG specific arguments')
@argument('--noisolation', help=textwrap.dedent('\n        Disable SVG isolation in the Notebook. By default, SVGs are isolated to\n        avoid namespace collisions between figures. Disabling SVG isolation\n        allows to reference previous figures or share CSS rules across a set\n        of SVGs.'), action='store_false', default=True, dest='isolate_svgs')
@argument_group('PNG', 'PNG specific arguments')
@argument('-u', '--units', type=str, choices=['px', 'in', 'cm', 'mm'], help=textwrap.dedent('\n        Units of png plotting device sent as an argument to *png* in R. One of\n        ["px", "in", "cm", "mm"].'))
@argument('-r', '--res', type=int, help=textwrap.dedent('\n        Resolution of png plotting device sent as an argument to *png* in R.\n        Defaults to 72 if *units* is one of ["in", "cm", "mm"].'))
@argument('--type', type=str, choices=['cairo', 'cairo-png', 'Xlib', 'quartz'], help=textwrap.dedent('\n        Type device used to generate the figure.\n        '))
@argument('-c', '--converter', default=None, help=textwrap.dedent("\n        Name of local converter to use. A converter contains the rules to\n        convert objects back and forth between Python and R. If not\n        specified/None, the defaut converter for the magic's module is used\n        (that is rpy2's default converter + numpy converter + pandas converter\n        if all three are available)."))
@argument('-d', '--display', default=None, help=textwrap.dedent('\n        Name of function to use to display the output of an R cell (the last\n        object or function call that does not have a left-hand side\n        assignment). That function will have the signature `(robject, args)`\n        where `robject` is the R objects that is an output of the cell, and\n        `args` a namespace with all parameters passed to the cell.\n        '))
@argument('code', nargs='*')
@needs_local_scope
@line_cell_magic
@no_var_expand
def R(self, line, cell=None, local_ns=None):
    """
        Execute code in R, optionally returning results to the Python runtime.

        In line mode, this will evaluate an expression and convert the returned
        value to a Python object.  The return value is determined by rpy2's
        behaviour of returning the result of evaluating the final expression.

        Multiple R expressions can be executed by joining them with
        semicolons::

            In [9]: %R X=c(1,4,5,7); sd(X); mean(X)
            Out[9]: array([ 4.25])

        In cell mode, this will run a block of R code. The resulting value
        is printed if it would be printed when evaluating the same code
        within a standard R REPL.

        Nothing is returned to python by default in cell mode::

            In [10]: %%R
               ....: Y = c(2,4,3,9)
               ....: summary(lm(Y~X))

            Call:
            lm(formula = Y ~ X)

            Residuals:
                1     2     3     4
             0.88 -0.24 -2.28  1.64

            Coefficients:
                        Estimate Std. Error t value Pr(>|t|)
            (Intercept)   0.0800     2.3000   0.035    0.975
            X             1.0400     0.4822   2.157    0.164

            Residual standard error: 2.088 on 2 degrees of freedom
            Multiple R-squared: 0.6993,Adjusted R-squared: 0.549
            F-statistic: 4.651 on 1 and 2 DF,  p-value: 0.1638

        In the notebook, plots are published as the output of the cell::

            %R plot(X, Y)

        will create a scatter plot of X bs Y.

        If cell is not None and line has some R code, it is prepended to
        the R code in cell.

        Objects can be passed back and forth between rpy2 and python via the
        -i -o flags in line::

            In [14]: Z = np.array([1,4,5,10])

            In [15]: %R -i Z mean(Z)
            Out[15]: array([ 5.])


            In [16]: %R -o W W=Z*mean(Z)
            Out[16]: array([  5.,  20.,  25.,  50.])

            In [17]: W
            Out[17]: array([  5.,  20.,  25.,  50.])

        The return value is determined by these rules:

        * If the cell is not None (i.e., has contents), the magic returns None.

        * If the final line results in a NULL value when evaluated
          by rpy2, then None is returned.

        * No attempt is made to convert the final value to a structured array.
          Use %Rget to push a structured array.

        * If the -n flag is present, there is no return value.

        * A trailing ';' will also result in no return value as the last
          value in the line is an empty string.
        """
    args = parse_argstring(self.R, line)
    if cell is None:
        code = ''
        return_output = True
        line_mode = True
    else:
        code = cell
        return_output = False
        line_mode = False
    code = ' '.join(args.code) + code
    if local_ns is None:
        local_ns = {}
    converter = self._find_converter(args.converter, local_ns)
    if args.input:
        with localconverter(converter) as cv:
            for arg in ','.join(args.input).split(','):
                self._import_name_into_r(arg, ro.globalenv, local_ns)
    if args.display:
        try:
            cell_display = local_ns[args.display]
        except KeyError:
            try:
                cell_display = self.shell.user_ns[args.display]
            except KeyError:
                raise NameError("name '%s' is not defined" % args.display)
    else:
        cell_display = CELL_DISPLAY_DEFAULT
    tmpd = self.setup_graphics(args)
    text_output = ''
    display_data = []
    try:
        if line_mode:
            for line in code.split(';'):
                text_result, result, visible = self.eval(line)
                text_output += text_result
            if text_result:
                return_output = False
        else:
            text_result, result, visible = self.eval(code)
            text_output += text_result
            if visible:
                with contextlib.ExitStack() as stack:
                    obj_in_module = rpy2.rinterface_lib.callbacks.obj_in_module
                    if self.cache_display_data:
                        stack.enter_context(obj_in_module(rpy2.rinterface_lib.callbacks, 'consolewrite_print', self.write_console_regular))
                    stack.enter_context(obj_in_module(rpy2.rinterface_lib.callbacks, 'consolewrite_warnerror', self.write_console_regular))
                    stack.enter_context(obj_in_module(rpy2.rinterface_lib.callbacks, '_WRITECONSOLE_EXCEPTION_LOG', '%s'))
                    cell_display(result, args)
                    text_output += self.flush()
    except RInterpreterError as e:
        print(e.stdout)
        if not e.stdout.endswith(e.err):
            print(e.err)
        raise e
    finally:
        if self.device in DEVICES_STATIC:
            ro.r('dev.off()')
        if text_output:
            displaypub.publish_display_data(source='RMagic.R', data={'text/plain': text_output})
        if self.device in DEVICES_STATIC_RASTER:
            for _ in display_figures(tmpd, format=self.device):
                if self.cache_display_data:
                    display_data.append(_)
        elif self.device == 'svg':
            for _ in display_figures_svg(tmpd):
                if self.cache_display_data:
                    display_data.append(_)
        if tmpd:
            rmtree(tmpd)
    if args.output:
        with localconverter(converter) as cv:
            for output in ','.join(args.output).split(','):
                output_ipy = ro.globalenv.find(output)
                self.shell.push({output: output_ipy})
    if self.cache_display_data:
        self.display_cache = display_data
    if return_output and (not args.noreturn):
        if result is not ri.NULL:
            with localconverter(converter) as cv:
                res = cv.rpy2py(result)
            return res