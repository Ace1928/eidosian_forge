from .basestemmer import BaseStemmer
from .among import Among
def __r_cyr_to_lat(self):
    v_1 = self.cursor
    try:
        while True:
            v_2 = self.cursor
            try:
                try:
                    while True:
                        v_3 = self.cursor
                        try:
                            self.bra = self.cursor
                            among_var = self.find_among(SerbianStemmer.a_0)
                            if among_var == 0:
                                raise lab3()
                            self.ket = self.cursor
                            if among_var == 1:
                                if not self.slice_from(u'a'):
                                    return False
                            elif among_var == 2:
                                if not self.slice_from(u'b'):
                                    return False
                            elif among_var == 3:
                                if not self.slice_from(u'v'):
                                    return False
                            elif among_var == 4:
                                if not self.slice_from(u'g'):
                                    return False
                            elif among_var == 5:
                                if not self.slice_from(u'd'):
                                    return False
                            elif among_var == 6:
                                if not self.slice_from(u'đ'):
                                    return False
                            elif among_var == 7:
                                if not self.slice_from(u'e'):
                                    return False
                            elif among_var == 8:
                                if not self.slice_from(u'ž'):
                                    return False
                            elif among_var == 9:
                                if not self.slice_from(u'z'):
                                    return False
                            elif among_var == 10:
                                if not self.slice_from(u'i'):
                                    return False
                            elif among_var == 11:
                                if not self.slice_from(u'j'):
                                    return False
                            elif among_var == 12:
                                if not self.slice_from(u'k'):
                                    return False
                            elif among_var == 13:
                                if not self.slice_from(u'l'):
                                    return False
                            elif among_var == 14:
                                if not self.slice_from(u'lj'):
                                    return False
                            elif among_var == 15:
                                if not self.slice_from(u'm'):
                                    return False
                            elif among_var == 16:
                                if not self.slice_from(u'n'):
                                    return False
                            elif among_var == 17:
                                if not self.slice_from(u'nj'):
                                    return False
                            elif among_var == 18:
                                if not self.slice_from(u'o'):
                                    return False
                            elif among_var == 19:
                                if not self.slice_from(u'p'):
                                    return False
                            elif among_var == 20:
                                if not self.slice_from(u'r'):
                                    return False
                            elif among_var == 21:
                                if not self.slice_from(u's'):
                                    return False
                            elif among_var == 22:
                                if not self.slice_from(u't'):
                                    return False
                            elif among_var == 23:
                                if not self.slice_from(u'ć'):
                                    return False
                            elif among_var == 24:
                                if not self.slice_from(u'u'):
                                    return False
                            elif among_var == 25:
                                if not self.slice_from(u'f'):
                                    return False
                            elif among_var == 26:
                                if not self.slice_from(u'h'):
                                    return False
                            elif among_var == 27:
                                if not self.slice_from(u'c'):
                                    return False
                            elif among_var == 28:
                                if not self.slice_from(u'č'):
                                    return False
                            elif among_var == 29:
                                if not self.slice_from(u'dž'):
                                    return False
                            elif not self.slice_from(u'š'):
                                return False
                            self.cursor = v_3
                            raise lab2()
                        except lab3:
                            pass
                        self.cursor = v_3
                        if self.cursor >= self.limit:
                            raise lab1()
                        self.cursor += 1
                except lab2:
                    pass
                continue
            except lab1:
                pass
            self.cursor = v_2
            break
    except lab0:
        pass
    self.cursor = v_1
    return True