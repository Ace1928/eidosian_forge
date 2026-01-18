from .basestemmer import BaseStemmer
from .among import Among
def __r_append_U_to_stems_ending_with_d_or_g(self):
    v_1 = self.limit - self.cursor
    try:
        v_2 = self.limit - self.cursor
        try:
            if not self.eq_s_b(u'd'):
                raise lab1()
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_2
        if not self.eq_s_b(u'g'):
            return False
    except lab0:
        pass
    self.cursor = self.limit - v_1
    try:
        v_3 = self.limit - self.cursor
        try:
            v_4 = self.limit - self.cursor
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel, 97, 305):
                raise lab3()
            try:
                v_5 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'a'):
                        raise lab5()
                    raise lab4()
                except lab5:
                    pass
                self.cursor = self.limit - v_5
                if not self.eq_s_b(u'ı'):
                    raise lab3()
            except lab4:
                pass
            self.cursor = self.limit - v_4
            c = self.cursor
            self.insert(self.cursor, self.cursor, u'ı')
            self.cursor = c
            raise lab2()
        except lab3:
            pass
        self.cursor = self.limit - v_3
        try:
            v_6 = self.limit - self.cursor
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel, 97, 305):
                raise lab6()
            try:
                v_7 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'e'):
                        raise lab8()
                    raise lab7()
                except lab8:
                    pass
                self.cursor = self.limit - v_7
                if not self.eq_s_b(u'i'):
                    raise lab6()
            except lab7:
                pass
            self.cursor = self.limit - v_6
            c = self.cursor
            self.insert(self.cursor, self.cursor, u'i')
            self.cursor = c
            raise lab2()
        except lab6:
            pass
        self.cursor = self.limit - v_3
        try:
            v_8 = self.limit - self.cursor
            if not self.go_out_grouping_b(TurkishStemmer.g_vowel, 97, 305):
                raise lab9()
            try:
                v_9 = self.limit - self.cursor
                try:
                    if not self.eq_s_b(u'o'):
                        raise lab11()
                    raise lab10()
                except lab11:
                    pass
                self.cursor = self.limit - v_9
                if not self.eq_s_b(u'u'):
                    raise lab9()
            except lab10:
                pass
            self.cursor = self.limit - v_8
            c = self.cursor
            self.insert(self.cursor, self.cursor, u'u')
            self.cursor = c
            raise lab2()
        except lab9:
            pass
        self.cursor = self.limit - v_3
        v_10 = self.limit - self.cursor
        if not self.go_out_grouping_b(TurkishStemmer.g_vowel, 97, 305):
            return False
        try:
            v_11 = self.limit - self.cursor
            try:
                if not self.eq_s_b(u'ö'):
                    raise lab13()
                raise lab12()
            except lab13:
                pass
            self.cursor = self.limit - v_11
            if not self.eq_s_b(u'ü'):
                return False
        except lab12:
            pass
        self.cursor = self.limit - v_10
        c = self.cursor
        self.insert(self.cursor, self.cursor, u'ü')
        self.cursor = c
    except lab2:
        pass
    return True