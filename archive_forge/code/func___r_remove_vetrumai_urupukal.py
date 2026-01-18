from .basestemmer import BaseStemmer
from .among import Among
def __r_remove_vetrumai_urupukal(self):
    self.B_found_a_match = False
    self.B_found_vetrumai_urupu = False
    if not self.__r_has_min_length():
        return False
    self.limit_backward = self.cursor
    self.cursor = self.limit
    try:
        v_1 = self.limit - self.cursor
        try:
            v_2 = self.limit - self.cursor
            self.ket = self.cursor
            if not self.eq_s_b(u'னை'):
                raise lab1()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            self.cursor = self.limit - v_2
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            v_3 = self.limit - self.cursor
            self.ket = self.cursor
            try:
                v_4 = self.limit - self.cursor
                try:
                    try:
                        v_5 = self.limit - self.cursor
                        try:
                            if not self.eq_s_b(u'ினை'):
                                raise lab6()
                            raise lab5()
                        except lab6:
                            pass
                        self.cursor = self.limit - v_5
                        if not self.eq_s_b(u'ை'):
                            raise lab4()
                    except lab5:
                        pass
                    v_6 = self.limit - self.cursor
                    v_7 = self.limit - self.cursor
                    try:
                        if self.find_among_b(TamilStemmer.a_18) == 0:
                            raise lab7()
                        raise lab4()
                    except lab7:
                        pass
                    self.cursor = self.limit - v_7
                    self.cursor = self.limit - v_6
                    raise lab3()
                except lab4:
                    pass
                self.cursor = self.limit - v_4
                if not self.eq_s_b(u'ை'):
                    raise lab2()
                v_8 = self.limit - self.cursor
                if self.find_among_b(TamilStemmer.a_19) == 0:
                    raise lab2()
                if not self.eq_s_b(u'்'):
                    raise lab2()
                self.cursor = self.limit - v_8
            except lab3:
                pass
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            self.cursor = self.limit - v_3
            raise lab0()
        except lab2:
            pass
        self.cursor = self.limit - v_1
        try:
            v_9 = self.limit - self.cursor
            self.ket = self.cursor
            try:
                v_10 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'ொடு'):
                        raise lab10()
                    raise lab9()
                except lab10:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ோடு'):
                        raise lab11()
                    raise lab9()
                except lab11:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ில்'):
                        raise lab12()
                    raise lab9()
                except lab12:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ிற்'):
                        raise lab13()
                    raise lab9()
                except lab13:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ின்'):
                        raise lab14()
                    v_11 = self.limit - self.cursor
                    v_12 = self.limit - self.cursor
                    try:
                        if not self.eq_s_b(u'ம'):
                            raise lab15()
                        raise lab14()
                    except lab15:
                        pass
                    self.cursor = self.limit - v_12
                    self.cursor = self.limit - v_11
                    raise lab9()
                except lab14:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ின்று'):
                        raise lab16()
                    raise lab9()
                except lab16:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ிருந்து'):
                        raise lab17()
                    raise lab9()
                except lab17:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'விட'):
                        raise lab18()
                    raise lab9()
                except lab18:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not len(self.current) >= 7:
                        raise lab19()
                    if not self.eq_s_b(u'ிடம்'):
                        raise lab19()
                    raise lab9()
                except lab19:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ால்'):
                        raise lab20()
                    raise lab9()
                except lab20:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ுடை'):
                        raise lab21()
                    raise lab9()
                except lab21:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ாமல்'):
                        raise lab22()
                    raise lab9()
                except lab22:
                    pass
                self.cursor = self.limit - v_10
                try:
                    if not self.eq_s_b(u'ல்'):
                        raise lab23()
                    v_13 = self.limit - self.cursor
                    v_14 = self.limit - self.cursor
                    try:
                        if self.find_among_b(TamilStemmer.a_20) == 0:
                            raise lab24()
                        raise lab23()
                    except lab24:
                        pass
                    self.cursor = self.limit - v_14
                    self.cursor = self.limit - v_13
                    raise lab9()
                except lab23:
                    pass
                self.cursor = self.limit - v_10
                if not self.eq_s_b(u'ுள்'):
                    raise lab8()
            except lab9:
                pass
            self.bra = self.cursor
            if not self.slice_from(u'்'):
                return False
            self.cursor = self.limit - v_9
            raise lab0()
        except lab8:
            pass
        self.cursor = self.limit - v_1
        try:
            v_15 = self.limit - self.cursor
            self.ket = self.cursor
            try:
                v_16 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'கண்'):
                        raise lab27()
                    raise lab26()
                except lab27:
                    pass
                self.cursor = self.limit - v_16
                try:
                    if not self.eq_s_b(u'முன்'):
                        raise lab28()
                    raise lab26()
                except lab28:
                    pass
                self.cursor = self.limit - v_16
                try:
                    if not self.eq_s_b(u'மேல்'):
                        raise lab29()
                    raise lab26()
                except lab29:
                    pass
                self.cursor = self.limit - v_16
                try:
                    if not self.eq_s_b(u'மேற்'):
                        raise lab30()
                    raise lab26()
                except lab30:
                    pass
                self.cursor = self.limit - v_16
                try:
                    if not self.eq_s_b(u'கீழ்'):
                        raise lab31()
                    raise lab26()
                except lab31:
                    pass
                self.cursor = self.limit - v_16
                try:
                    if not self.eq_s_b(u'பின்'):
                        raise lab32()
                    raise lab26()
                except lab32:
                    pass
                self.cursor = self.limit - v_16
                if not self.eq_s_b(u'து'):
                    raise lab25()
                v_17 = self.limit - self.cursor
                v_18 = self.limit - self.cursor
                try:
                    if self.find_among_b(TamilStemmer.a_21) == 0:
                        raise lab33()
                    raise lab25()
                except lab33:
                    pass
                self.cursor = self.limit - v_18
                self.cursor = self.limit - v_17
            except lab26:
                pass
            self.bra = self.cursor
            if not self.slice_del():
                return False
            self.cursor = self.limit - v_15
            raise lab0()
        except lab25:
            pass
        self.cursor = self.limit - v_1
        v_19 = self.limit - self.cursor
        self.ket = self.cursor
        if not self.eq_s_b(u'ீ'):
            return False
        self.bra = self.cursor
        if not self.slice_from(u'ி'):
            return False
        self.cursor = self.limit - v_19
    except lab0:
        pass
    self.B_found_a_match = True
    self.B_found_vetrumai_urupu = True
    v_20 = self.limit - self.cursor
    try:
        self.ket = self.cursor
        if not self.eq_s_b(u'ின்'):
            raise lab34()
        self.bra = self.cursor
        if not self.slice_from(u'்'):
            return False
    except lab34:
        pass
    self.cursor = self.limit - v_20
    self.cursor = self.limit_backward
    self.__r_fix_endings()
    return True