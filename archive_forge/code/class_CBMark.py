from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
class CBMark:
    opNames = 'm l c h'.split()
    opCount = (1, 1, 3, 0)

    def __init__(self, ops, points, bounds, slack=0.05):
        self.ops = ops
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.points = points
        self.slack = slack

    def scaledRender(self, size, ds=0):
        """
        >>> print(cbmarks['check'].scaledRender(20))
        12.97075 14.68802 m 15.00139 17.16992 l 15.9039 18.1727 17.93454 18.67409 19.2883 18.67409 c 19.46379 18.27298 l 17.13231 15.51532 l 11.91783 8.62117 l 8.307799 3.030641 l 7.430362 1.526462 l 7.305014 1.275766 7.154596 .97493 6.9039 .824513 c 6.577994 .674095 5.825905 .674095 5.47493 .674095 c 4.672702 .674095 4.497214 .674095 4.321727 .799443 c 4.071031 .97493 3.945682 1.325905 3.770195 1.67688 c 3.218663 2.830084 2.240947 5.337047 2.240947 6.590529 c 2.240947 7.016713 2.491643 7.21727 2.817549 7.442897 c 3.344011 7.818942 4.0961 8.245125 4.747911 8.245125 c 5.249304 8.245125 5.299443 7.818942 5.449861 7.417827 c 5.951253 6.239554 l 6.026462 6.038997 6.252089 5.337047 6.527855 5.337047 c 6.778552 5.337047 7.079387 5.913649 7.179666 6.089136 c 12.97075 14.68802 l h f
        >>> print(cbmarks['cross'].scaledRender(20))
        19.9104 17.43931 m 12.41908 10 l 19.9104 2.534682 l 18.37572 1 l 10.9104 8.491329 l 3.445087 1 l 1.910405 2.534682 l 9.427746 10 l 1.910405 17.46532 l 3.445087 19 l 10.9104 11.50867 l 18.37572 19 l 19.9104 17.43931 l h f
        >>> print(cbmarks['circle'].scaledRender(20))
        1.872576 9.663435 m 1.872576 14.64958 5.936288 18.61357 10.89751 18.61357 c 15.8338 18.61357 19.87258 14.59972 19.87258 9.663435 c 19.87258 4.727147 15.8338 .688366 10.89751 .688366 c 5.936288 .688366 1.872576 4.677285 1.872576 9.663435 c h f
        >>> print(cbmarks['star'].scaledRender(20))
        10.85542 18.3253 m 12.90361 11.84337 l 19.84337 11.84337 l 14.25301 7.650602 l 16.42169 1 l 10.85542 5.096386 l 5.289157 1 l 7.481928 7.650602 l 1.843373 11.84337 l 8.759036 11.84337 l 10.85542 18.3253 l h f
        >>> print(cbmarks['diamond'].scaledRender(20))
        17.43533 9.662031 m 15.63282 7.484006 l 10.85118 .649513 l 8.422809 4.329624 l 5.919332 7.659249 l 4.267038 9.662031 l 6.16968 12.0153 l 10.85118 18.64951 l 12.75382 15.4701 15.00695 12.49096 17.43533 9.662031 c h f
        """
        W = H = size - 2 * ds
        xmin = self.xmin
        ymin = self.ymin
        w = self.xmax - xmin
        h = self.ymax - ymin
        slack = self.slack * min(W, H)
        sx = (W - 2 * slack) / float(w)
        sy = (H - 2 * slack) / float(h)
        sx = sy = min(sx, sy)
        w *= sx
        h *= sy
        dx = ds + (W - w) * 0.5
        dy = ds + (H - h) * 0.5
        xsc = lambda v: fp_str((v - xmin) * sx + dx)
        ysc = lambda v: fp_str((v - ymin) * sy + dy)
        opNames = self.opNames
        opCount = self.opCount
        C = [].append
        i = 0
        points = self.points
        for op in self.ops:
            c = opCount[op]
            for _ in range(c):
                C(xsc(points[i]))
                C(ysc(points[i + 1]))
                i += 2
            C(opNames[op])
        C('f')
        return ' '.join(C.__self__)