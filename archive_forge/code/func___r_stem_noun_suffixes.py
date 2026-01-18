from .basestemmer import BaseStemmer
from .among import Among
def __r_stem_noun_suffixes(self):
    try:
        v_1 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            if not self.__r_mark_lAr():
                raise lab1()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_2 = self.limit - self.cursor
            try:
                if not self.__r_stem_suffix_chain_before_ki():
                    self.cursor = self.limit - v_2
                    raise lab2()
            except lab2:
                pass
            raise lab0()
        except lab1:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.__r_mark_ncA():
                raise lab3()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_3 = self.limit - self.cursor
            try:
                try:
                    v_4 = self.limit - self.cursor
                    try:
                        self.ket = self.cursor
                        if not self.__r_mark_lArI():
                            raise lab6()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        raise lab5()
                    except lab6:
                        pass
                    self.cursor = self.limit - v_4
                    try:
                        self.ket = self.cursor
                        try:
                            v_5 = self.limit - self.cursor
                            try:
                                if not self.__r_mark_possessives():
                                    raise lab9()
                                raise lab8()
                            except lab9:
                                pass
                            self.cursor = self.limit - v_5
                            if not self.__r_mark_sU():
                                raise lab7()
                        except lab8:
                            pass
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_6 = self.limit - self.cursor
                        try:
                            self.ket = self.cursor
                            if not self.__r_mark_lAr():
                                self.cursor = self.limit - v_6
                                raise lab10()
                            self.bra = self.cursor
                            if not self.slice_del():
                                return False
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_6
                                raise lab10()
                        except lab10:
                            pass
                        raise lab5()
                    except lab7:
                        pass
                    self.cursor = self.limit - v_4
                    self.ket = self.cursor
                    if not self.__r_mark_lAr():
                        self.cursor = self.limit - v_3
                        raise lab4()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    if not self.__r_stem_suffix_chain_before_ki():
                        self.cursor = self.limit - v_3
                        raise lab4()
                except lab5:
                    pass
            except lab4:
                pass
            raise lab0()
        except lab3:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_7 = self.limit - self.cursor
                try:
                    if not self.__r_mark_ndA():
                        raise lab13()
                    raise lab12()
                except lab13:
                    pass
                self.cursor = self.limit - v_7
                if not self.__r_mark_nA():
                    raise lab11()
            except lab12:
                pass
            try:
                v_8 = self.limit - self.cursor
                try:
                    if not self.__r_mark_lArI():
                        raise lab15()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    raise lab14()
                except lab15:
                    pass
                self.cursor = self.limit - v_8
                try:
                    if not self.__r_mark_sU():
                        raise lab16()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    v_9 = self.limit - self.cursor
                    try:
                        self.ket = self.cursor
                        if not self.__r_mark_lAr():
                            self.cursor = self.limit - v_9
                            raise lab17()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        if not self.__r_stem_suffix_chain_before_ki():
                            self.cursor = self.limit - v_9
                            raise lab17()
                    except lab17:
                        pass
                    raise lab14()
                except lab16:
                    pass
                self.cursor = self.limit - v_8
                if not self.__r_stem_suffix_chain_before_ki():
                    raise lab11()
            except lab14:
                pass
            raise lab0()
        except lab11:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_10 = self.limit - self.cursor
                try:
                    if not self.__r_mark_ndAn():
                        raise lab20()
                    raise lab19()
                except lab20:
                    pass
                self.cursor = self.limit - v_10
                if not self.__r_mark_nU():
                    raise lab18()
            except lab19:
                pass
            try:
                v_11 = self.limit - self.cursor
                try:
                    if not self.__r_mark_sU():
                        raise lab22()
                    self.bra = self.cursor
                    if not self.slice_del():
                        return False
                    v_12 = self.limit - self.cursor
                    try:
                        self.ket = self.cursor
                        if not self.__r_mark_lAr():
                            self.cursor = self.limit - v_12
                            raise lab23()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        if not self.__r_stem_suffix_chain_before_ki():
                            self.cursor = self.limit - v_12
                            raise lab23()
                    except lab23:
                        pass
                    raise lab21()
                except lab22:
                    pass
                self.cursor = self.limit - v_11
                if not self.__r_mark_lArI():
                    raise lab18()
            except lab21:
                pass
            raise lab0()
        except lab18:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.__r_mark_DAn():
                raise lab24()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_13 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                try:
                    v_14 = self.limit - self.cursor
                    try:
                        if not self.__r_mark_possessives():
                            raise lab27()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_15 = self.limit - self.cursor
                        try:
                            self.ket = self.cursor
                            if not self.__r_mark_lAr():
                                self.cursor = self.limit - v_15
                                raise lab28()
                            self.bra = self.cursor
                            if not self.slice_del():
                                return False
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_15
                                raise lab28()
                        except lab28:
                            pass
                        raise lab26()
                    except lab27:
                        pass
                    self.cursor = self.limit - v_14
                    try:
                        if not self.__r_mark_lAr():
                            raise lab29()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_16 = self.limit - self.cursor
                        try:
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_16
                                raise lab30()
                        except lab30:
                            pass
                        raise lab26()
                    except lab29:
                        pass
                    self.cursor = self.limit - v_14
                    if not self.__r_stem_suffix_chain_before_ki():
                        self.cursor = self.limit - v_13
                        raise lab25()
                except lab26:
                    pass
            except lab25:
                pass
            raise lab0()
        except lab24:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_17 = self.limit - self.cursor
                try:
                    if not self.__r_mark_nUn():
                        raise lab33()
                    raise lab32()
                except lab33:
                    pass
                self.cursor = self.limit - v_17
                if not self.__r_mark_ylA():
                    raise lab31()
            except lab32:
                pass
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_18 = self.limit - self.cursor
            try:
                try:
                    v_19 = self.limit - self.cursor
                    try:
                        self.ket = self.cursor
                        if not self.__r_mark_lAr():
                            raise lab36()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        if not self.__r_stem_suffix_chain_before_ki():
                            raise lab36()
                        raise lab35()
                    except lab36:
                        pass
                    self.cursor = self.limit - v_19
                    try:
                        self.ket = self.cursor
                        try:
                            v_20 = self.limit - self.cursor
                            try:
                                if not self.__r_mark_possessives():
                                    raise lab39()
                                raise lab38()
                            except lab39:
                                pass
                            self.cursor = self.limit - v_20
                            if not self.__r_mark_sU():
                                raise lab37()
                        except lab38:
                            pass
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_21 = self.limit - self.cursor
                        try:
                            self.ket = self.cursor
                            if not self.__r_mark_lAr():
                                self.cursor = self.limit - v_21
                                raise lab40()
                            self.bra = self.cursor
                            if not self.slice_del():
                                return False
                            if not self.__r_stem_suffix_chain_before_ki():
                                self.cursor = self.limit - v_21
                                raise lab40()
                        except lab40:
                            pass
                        raise lab35()
                    except lab37:
                        pass
                    self.cursor = self.limit - v_19
                    if not self.__r_stem_suffix_chain_before_ki():
                        self.cursor = self.limit - v_18
                        raise lab34()
                except lab35:
                    pass
            except lab34:
                pass
            raise lab0()
        except lab31:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            if not self.__r_mark_lArI():
                raise lab41()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            raise lab0()
        except lab41:
            pass
        self.cursor = self.limit - v_1
        try:
            if not self.__r_stem_suffix_chain_before_ki():
                raise lab42()
            raise lab0()
        except lab42:
            pass
        self.cursor = self.limit - v_1
        try:
            self.ket = self.cursor
            try:
                v_22 = self.limit - self.cursor
                try:
                    if not self.__r_mark_DA():
                        raise lab45()
                    raise lab44()
                except lab45:
                    pass
                self.cursor = self.limit - v_22
                try:
                    if not self.__r_mark_yU():
                        raise lab46()
                    raise lab44()
                except lab46:
                    pass
                self.cursor = self.limit - v_22
                if not self.__r_mark_yA():
                    raise lab43()
            except lab44:
                pass
            self.bra = self.cursor
            if not self.slice_del():
                return False
            v_23 = self.limit - self.cursor
            try:
                self.ket = self.cursor
                try:
                    v_24 = self.limit - self.cursor
                    try:
                        if not self.__r_mark_possessives():
                            raise lab49()
                        self.bra = self.cursor
                        if not self.slice_del():
                            return False
                        v_25 = self.limit - self.cursor
                        try:
                            self.ket = self.cursor
                            if not self.__r_mark_lAr():
                                self.cursor = self.limit - v_25
                                raise lab50()
                        except lab50:
                            pass
                        raise lab48()
                    except lab49:
                        pass
                    self.cursor = self.limit - v_24
                    if not self.__r_mark_lAr():
                        self.cursor = self.limit - v_23
                        raise lab47()
                except lab48:
                    pass
                self.bra = self.cursor
                if not self.slice_del():
                    return False
                self.ket = self.cursor
                if not self.__r_stem_suffix_chain_before_ki():
                    self.cursor = self.limit - v_23
                    raise lab47()
            except lab47:
                pass
            raise lab0()
        except lab43:
            pass
        self.cursor = self.limit - v_1
        self.ket = self.cursor
        try:
            v_26 = self.limit - self.cursor
            try:
                if not self.__r_mark_possessives():
                    raise lab52()
                raise lab51()
            except lab52:
                pass
            self.cursor = self.limit - v_26
            if not self.__r_mark_sU():
                return False
        except lab51:
            pass
        self.bra = self.cursor
        if not self.slice_del():
            return False
        v_27 = self.limit - self.cursor
        try:
            self.ket = self.cursor
            if not self.__r_mark_lAr():
                self.cursor = self.limit - v_27
                raise lab53()
            self.bra = self.cursor
            if not self.slice_del():
                return False
            if not self.__r_stem_suffix_chain_before_ki():
                self.cursor = self.limit - v_27
                raise lab53()
        except lab53:
            pass
    except lab0:
        pass
    return True