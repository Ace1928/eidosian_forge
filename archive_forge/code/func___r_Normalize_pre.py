from .basestemmer import BaseStemmer
from .among import Among
def __r_Normalize_pre(self):
    v_1 = self.cursor
    try:
        while True:
            v_2 = self.cursor
            try:
                try:
                    v_3 = self.cursor
                    try:
                        self.bra = self.cursor
                        among_var = self.find_among(ArabicStemmer.a_0)
                        if among_var == 0:
                            raise lab3()
                        self.ket = self.cursor
                        if among_var == 1:
                            if not self.slice_del():
                                return False
                        elif among_var == 2:
                            if not self.slice_from(u'0'):
                                return False
                        elif among_var == 3:
                            if not self.slice_from(u'1'):
                                return False
                        elif among_var == 4:
                            if not self.slice_from(u'2'):
                                return False
                        elif among_var == 5:
                            if not self.slice_from(u'3'):
                                return False
                        elif among_var == 6:
                            if not self.slice_from(u'4'):
                                return False
                        elif among_var == 7:
                            if not self.slice_from(u'5'):
                                return False
                        elif among_var == 8:
                            if not self.slice_from(u'6'):
                                return False
                        elif among_var == 9:
                            if not self.slice_from(u'7'):
                                return False
                        elif among_var == 10:
                            if not self.slice_from(u'8'):
                                return False
                        elif among_var == 11:
                            if not self.slice_from(u'9'):
                                return False
                        elif among_var == 12:
                            if not self.slice_from(u'ء'):
                                return False
                        elif among_var == 13:
                            if not self.slice_from(u'أ'):
                                return False
                        elif among_var == 14:
                            if not self.slice_from(u'إ'):
                                return False
                        elif among_var == 15:
                            if not self.slice_from(u'ئ'):
                                return False
                        elif among_var == 16:
                            if not self.slice_from(u'آ'):
                                return False
                        elif among_var == 17:
                            if not self.slice_from(u'ؤ'):
                                return False
                        elif among_var == 18:
                            if not self.slice_from(u'ا'):
                                return False
                        elif among_var == 19:
                            if not self.slice_from(u'ب'):
                                return False
                        elif among_var == 20:
                            if not self.slice_from(u'ة'):
                                return False
                        elif among_var == 21:
                            if not self.slice_from(u'ت'):
                                return False
                        elif among_var == 22:
                            if not self.slice_from(u'ث'):
                                return False
                        elif among_var == 23:
                            if not self.slice_from(u'ج'):
                                return False
                        elif among_var == 24:
                            if not self.slice_from(u'ح'):
                                return False
                        elif among_var == 25:
                            if not self.slice_from(u'خ'):
                                return False
                        elif among_var == 26:
                            if not self.slice_from(u'د'):
                                return False
                        elif among_var == 27:
                            if not self.slice_from(u'ذ'):
                                return False
                        elif among_var == 28:
                            if not self.slice_from(u'ر'):
                                return False
                        elif among_var == 29:
                            if not self.slice_from(u'ز'):
                                return False
                        elif among_var == 30:
                            if not self.slice_from(u'س'):
                                return False
                        elif among_var == 31:
                            if not self.slice_from(u'ش'):
                                return False
                        elif among_var == 32:
                            if not self.slice_from(u'ص'):
                                return False
                        elif among_var == 33:
                            if not self.slice_from(u'ض'):
                                return False
                        elif among_var == 34:
                            if not self.slice_from(u'ط'):
                                return False
                        elif among_var == 35:
                            if not self.slice_from(u'ظ'):
                                return False
                        elif among_var == 36:
                            if not self.slice_from(u'ع'):
                                return False
                        elif among_var == 37:
                            if not self.slice_from(u'غ'):
                                return False
                        elif among_var == 38:
                            if not self.slice_from(u'ف'):
                                return False
                        elif among_var == 39:
                            if not self.slice_from(u'ق'):
                                return False
                        elif among_var == 40:
                            if not self.slice_from(u'ك'):
                                return False
                        elif among_var == 41:
                            if not self.slice_from(u'ل'):
                                return False
                        elif among_var == 42:
                            if not self.slice_from(u'م'):
                                return False
                        elif among_var == 43:
                            if not self.slice_from(u'ن'):
                                return False
                        elif among_var == 44:
                            if not self.slice_from(u'ه'):
                                return False
                        elif among_var == 45:
                            if not self.slice_from(u'و'):
                                return False
                        elif among_var == 46:
                            if not self.slice_from(u'ى'):
                                return False
                        elif among_var == 47:
                            if not self.slice_from(u'ي'):
                                return False
                        elif among_var == 48:
                            if not self.slice_from(u'لا'):
                                return False
                        elif among_var == 49:
                            if not self.slice_from(u'لأ'):
                                return False
                        elif among_var == 50:
                            if not self.slice_from(u'لإ'):
                                return False
                        elif not self.slice_from(u'لآ'):
                            return False
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