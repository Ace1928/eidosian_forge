from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_common_word_endings(self):
    self.B_found_a_match = False
    if not self.__r_has_min_length():
        return False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    try:
        v_1 = self.limit - self.cursor
        try:
            v_2 = self.limit - self.cursor
            self.ket = self.cursor
            try:
                v_3 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'ுடன்'):
                        raise lab3()
                    raise lab2()
                except lab3:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ில்லை'):
                        raise lab4()
                    raise lab2()
                except lab4:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ிடம்'):
                        raise lab5()
                    raise lab2()
                except lab5:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ின்றி'):
                        raise lab6()
                    raise lab2()
                except lab6:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ாகி'):
                        raise lab7()
                    raise lab2()
                except lab7:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ாகிய'):
                        raise lab8()
                    raise lab2()
                except lab8:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ென்று'):
                        raise lab9()
                    raise lab2()
                except lab9:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ுள்ள'):
                        raise lab10()
                    raise lab2()
                except lab10:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ுடைய'):
                        raise lab11()
                    raise lab2()
                except lab11:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ுடை'):
                        raise lab12()
                    raise lab2()
                except lab12:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ெனும்'):
                        raise lab13()
                    raise lab2()
                except lab13:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ல்ல'):
                        raise lab14()
                    v_4 = self.limit - self.cursor
                    v_5 = self.limit - self.cursor
                    try:
                        if self.find_among_b(TamilStemmer.a_16) == 0:
                            raise lab15()
                        raise lab14()
                    except lab15:
                        pass
                    self.cursor = self.limit - v_5
                    self.cursor = self.limit - v_4
                    raise lab2()
                except lab14:
                    pass
                self.cursor = self.limit - v_3
                try:
                    if not self.eq_s_b(u'ென'):
                        raise lab16()
                    raise lab2()
                except lab16:
                    pass
                self.cursor = self.limit - v_3
                if not self.eq_s_b(u'ாகி'):
                    raise lab1()
            except lab2:
                pass
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            self.B_found_a_match = True
            self.cursor = self.limit - v_2
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        v_6 = self.limit - self.cursor
        self.ket = self.cursor
        if self.find_among_b(TamilStemmer.a_17) == 0:
            return False
        self.bra = self.cursor
        if not self.slice_del():
            return False
        self.B_found_a_match = True
        self.cursor = self.limit - v_6
    except lab0:
        pass
    self.cursor = self.limit_backward
    self.__r_fix_endings()
    return True