from .basestemmer import BaseStemmer
from .among import Among
def __r_prelude(self):
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
                            among_var = self.find_among(YiddishStemmer.a_0)
                            if among_var == 0:
                                raise lab3()
                            self.ket = self.cursor
                            if among_var == 1:
                                v_4 = self.cursor
                                try:
                                    if not self.eq_s(u'ּ'):
                                        raise lab4()
                                    raise lab3()
                                except lab4:
                                    pass
                                self.cursor = v_4
                                if not self.slice_from(u'װ'):
                                    return False
                            elif among_var == 2:
                                v_5 = self.cursor
                                try:
                                    if not self.eq_s(u'ִ'):
                                        raise lab5()
                                    raise lab3()
                                except lab5:
                                    pass
                                self.cursor = v_5
                                if not self.slice_from(u'ױ'):
                                    return False
                            elif among_var == 3:
                                v_6 = self.cursor
                                try:
                                    if not self.eq_s(u'ִ'):
                                        raise lab6()
                                    raise lab3()
                                except lab6:
                                    pass
                                self.cursor = v_6
                                if not self.slice_from(u'ײ'):
                                    return False
                            elif among_var == 4:
                                if not self.slice_from(u'כ'):
                                    return False
                            elif among_var == 5:
                                if not self.slice_from(u'מ'):
                                    return False
                            elif among_var == 6:
                                if not self.slice_from(u'נ'):
                                    return False
                            elif among_var == 7:
                                if not self.slice_from(u'פ'):
                                    return False
                            elif not self.slice_from(u'צ'):
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
    v_7 = self.cursor
    try:
        while True:
            v_8 = self.cursor
            try:
                try:
                    while True:
                        v_9 = self.cursor
                        try:
                            self.bra = self.cursor
                            if not self.in_grouping(YiddishStemmer.g_niked, 1456, 1474):
                                raise lab10()
                            self.ket = self.cursor
                            if not self.slice_del():
                                return False
                            self.cursor = v_9
                            raise lab9()
                        except lab10:
                            pass
                        self.cursor = v_9
                        if self.cursor >= self.limit:
                            raise lab8()
                        self.cursor += 1
                except lab9:
                    pass
                continue
            except lab8:
                pass
            self.cursor = v_8
            break
    except lab7:
        pass
    self.cursor = v_7
    return True