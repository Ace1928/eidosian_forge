import sys
def Df_e(u, v, alpha=1.0, beta=0.0, trans='N'):
    if trans == 'N':
        Df(u[0], v, alpha=alpha, beta=beta, trans='N')
        v[0] -= alpha * u[1]
    else:
        Df(u, v[0], alpha=alpha, beta=beta, trans='T')
        v[1] = -alpha * u[0] + beta * v[1]