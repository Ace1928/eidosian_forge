import sys

    Solves a geometric program

        minimize    log sum exp (F0*x+g0)
        subject to  log sum exp (Fi*x+gi) <= 0,  i=1,...,m
                    G*x <= h      
                    A*x = b

    Input arguments.

        K is a list of positive integers [K0, K1, K2, ..., Km].

        F is a sum(K)xn dense or sparse 'd' matrix with block rows F0, 
        F1, ..., Fm.  Each Fi is Kixn.

        g is a sum(K)x1 dense or sparse 'd' matrix with blocks g0, g1, 
        g2, ..., gm.  Each gi is Kix1.

        G is an mxn dense or sparse 'd' matrix.

        h is an mx1 dense 'd' matrix.

        A is a pxn dense or sparse 'd' matrix.

        b is a px1 dense 'd' matrix.

        The default values for G, h, A and b are empty matrices with 
        zero rows.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'snl', 'sl',
        'znl', 'zl', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap', 'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If status is 'optimal', x, snl, sl, y, znl, zl  are approximate 
        solutions of the primal and dual optimality conditions

            f(x)[1:] + snl = 0,  G*x + sl = h,  A*x = b 
            Df(x)'*[1; znl] + G'*zl + A'*y + c = 0 
            snl >= 0,  znl >= 0,  sl >= 0,  zl >= 0
            snl'*znl + sl'* zl = 0,

        where fk(x) = log sum exp (Fk*x + gk). 

        If status is 'unknown', x, snl, sl, y, znl, zl are the last
        iterates before termination.  They satisfy snl > 0, znl > 0, 
        sl > 0, zl > 0, but are not necessarily feasible.

        The values of the other fields are the values returned by cpl()
        applied to the epigraph form problem

            minimize   t 
            subjec to  f0(x) <= t
                       fk(x) <= 0, k = 1, ..., mnl
                       G*x <= h
                       A*x = b.

        Termination with status 'unknown' indicates that the algorithm 
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, snl, sl, znl, zl are close to optimal.


    Control parameters.

       The following control parameters can be modified by adding an
       entry to the dictionary options.

       options['show_progress'] True/False (default: True)
       options['maxiters'] positive integer (default: 100)
       options['refinement'] nonnegative integer (default: 1)
       options['abstol'] scalar (default: 1e-7)
       options['reltol'] scalar (default: 1e-6)
       options['feastol'] scalar (default: 1e-7).
    