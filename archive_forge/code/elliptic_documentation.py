from .functions import defun, defun_wrapped

    Called with three arguments `n, \phi, m`, evaluates the Legendre
    incomplete elliptic integral of the third kind

    .. math ::

        \Pi(n; \phi, m) = \int_0^{\phi}
            \frac{dt}{(1-n \sin^2 t) \sqrt{1-m \sin^2 t}} =
            \int_0^{\sin \phi}
            \frac{dt}{(1-nt^2) \sqrt{1-t^2} \sqrt{1-mt^2}}.

    Called with two arguments `n, m`, evaluates the complete
    elliptic integral of the third kind
    `\Pi(n,m) = \Pi(n; \frac{\pi}{2},m)`.

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(\phi) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        \Pi(n,\phi+k\pi,m) = 2k\Pi(n,m) + \Pi(n,\phi,m), k \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellippi.py
    .. image :: /plots/ellippi.png

    **Examples for the complete integral**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellippi(0,-5); ellipk(-5)
        0.9555039270640439337379334
        0.9555039270640439337379334
        >>> ellippi(inf,2)
        0.0
        >>> ellippi(2,inf)
        0.0
        >>> abs(ellippi(1,5))
        +inf
        >>> abs(ellippi(0.25,1))
        +inf

    Evaluation in terms of simpler functions::

        >>> ellippi(0.25,0.25); ellipe(0.25)/(1-0.25)
        1.956616279119236207279727
        1.956616279119236207279727
        >>> ellippi(3,0); pi/(2*sqrt(-2))
        (0.0 - 1.11072073453959156175397j)
        (0.0 - 1.11072073453959156175397j)
        >>> ellippi(-3,0); pi/(2*sqrt(4))
        0.7853981633974483096156609
        0.7853981633974483096156609

    **Examples for the incomplete integral**

    Basic values and limits::

        >>> ellippi(0.25,-0.5); ellippi(0.25,pi/2,-0.5)
        1.622944760954741603710555
        1.622944760954741603710555
        >>> ellippi(1,0,1)
        0.0
        >>> ellippi(inf,0,1)
        0.0
        >>> ellippi(0,0.25,0.5); ellipf(0.25,0.5)
        0.2513040086544925794134591
        0.2513040086544925794134591
        >>> ellippi(1,1,1); (log(sec(1)+tan(1))+sec(1)*tan(1))/2
        2.054332933256248668692452
        2.054332933256248668692452
        >>> ellippi(0.25, 53*pi/2, 0.75); 53*ellippi(0.25,0.75)
        135.240868757890840755058
        135.240868757890840755058
        >>> ellippi(0.5,pi/4,0.5); 2*ellipe(pi/4,0.5)-1/sqrt(3)
        0.9190227391656969903987269
        0.9190227391656969903987269

    Complex arguments are supported::

        >>> ellippi(0.5, 5+6j-2*pi, -7-8j)
        (-0.3612856620076747660410167 + 0.5217735339984807829755815j)

    Some degenerate cases::

        >>> ellippi(1,1)
        +inf
        >>> ellippi(1,0)
        +inf
        >>> ellippi(1,2,0)
        +inf
        >>> ellippi(1,2,1)
        +inf
        >>> ellippi(1,0,1)
        0.0

    