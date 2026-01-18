from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
class PropertyValue_ReferenceValue(ProtocolBuffer.ProtocolMessage):
    has_app_ = 0
    app_ = ''
    has_name_space_ = 0
    name_space_ = ''
    has_database_id_ = 0
    database_id_ = ''

    def __init__(self, contents=None):
        self.pathelement_ = []
        if contents is not None:
            self.MergeFromString(contents)

    def app(self):
        return self.app_

    def set_app(self, x):
        self.has_app_ = 1
        self.app_ = x

    def clear_app(self):
        if self.has_app_:
            self.has_app_ = 0
            self.app_ = ''

    def has_app(self):
        return self.has_app_

    def name_space(self):
        return self.name_space_

    def set_name_space(self, x):
        self.has_name_space_ = 1
        self.name_space_ = x

    def clear_name_space(self):
        if self.has_name_space_:
            self.has_name_space_ = 0
            self.name_space_ = ''

    def has_name_space(self):
        return self.has_name_space_

    def pathelement_size(self):
        return len(self.pathelement_)

    def pathelement_list(self):
        return self.pathelement_

    def pathelement(self, i):
        return self.pathelement_[i]

    def mutable_pathelement(self, i):
        return self.pathelement_[i]

    def add_pathelement(self):
        x = PropertyValue_ReferenceValuePathElement()
        self.pathelement_.append(x)
        return x

    def clear_pathelement(self):
        self.pathelement_ = []

    def database_id(self):
        return self.database_id_

    def set_database_id(self, x):
        self.has_database_id_ = 1
        self.database_id_ = x

    def clear_database_id(self):
        if self.has_database_id_:
            self.has_database_id_ = 0
            self.database_id_ = ''

    def has_database_id(self):
        return self.has_database_id_

    def MergeFrom(self, x):
        assert x is not self
        if x.has_app():
            self.set_app(x.app())
        if x.has_name_space():
            self.set_name_space(x.name_space())
        for i in range(x.pathelement_size()):
            self.add_pathelement().CopyFrom(x.pathelement(i))
        if x.has_database_id():
            self.set_database_id(x.database_id())

    def Equals(self, x):
        if x is self:
            return 1
        if self.has_app_ != x.has_app_:
            return 0
        if self.has_app_ and self.app_ != x.app_:
            return 0
        if self.has_name_space_ != x.has_name_space_:
            return 0
        if self.has_name_space_ and self.name_space_ != x.name_space_:
            return 0
        if len(self.pathelement_) != len(x.pathelement_):
            return 0
        for e1, e2 in zip(self.pathelement_, x.pathelement_):
            if e1 != e2:
                return 0
        if self.has_database_id_ != x.has_database_id_:
            return 0
        if self.has_database_id_ and self.database_id_ != x.database_id_:
            return 0
        return 1

    def IsInitialized(self, debug_strs=None):
        initialized = 1
        if not self.has_app_:
            initialized = 0
            if debug_strs is not None:
                debug_strs.append('Required field: app not set.')
        for p in self.pathelement_:
            if not p.IsInitialized(debug_strs):
                initialized = 0
        return initialized

    def ByteSize(self):
        n = 0
        n += self.lengthString(len(self.app_))
        if self.has_name_space_:
            n += 2 + self.lengthString(len(self.name_space_))
        n += 2 * len(self.pathelement_)
        for i in range(len(self.pathelement_)):
            n += self.pathelement_[i].ByteSize()
        if self.has_database_id_:
            n += 2 + self.lengthString(len(self.database_id_))
        return n + 1

    def ByteSizePartial(self):
        n = 0
        if self.has_app_:
            n += 1
            n += self.lengthString(len(self.app_))
        if self.has_name_space_:
            n += 2 + self.lengthString(len(self.name_space_))
        n += 2 * len(self.pathelement_)
        for i in range(len(self.pathelement_)):
            n += self.pathelement_[i].ByteSizePartial()
        if self.has_database_id_:
            n += 2 + self.lengthString(len(self.database_id_))
        return n

    def Clear(self):
        self.clear_app()
        self.clear_name_space()
        self.clear_pathelement()
        self.clear_database_id()

    def OutputUnchecked(self, out):
        out.putVarInt32(106)
        out.putPrefixedString(self.app_)
        for i in range(len(self.pathelement_)):
            out.putVarInt32(115)
            self.pathelement_[i].OutputUnchecked(out)
            out.putVarInt32(116)
        if self.has_name_space_:
            out.putVarInt32(162)
            out.putPrefixedString(self.name_space_)
        if self.has_database_id_:
            out.putVarInt32(186)
            out.putPrefixedString(self.database_id_)

    def OutputPartial(self, out):
        if self.has_app_:
            out.putVarInt32(106)
            out.putPrefixedString(self.app_)
        for i in range(len(self.pathelement_)):
            out.putVarInt32(115)
            self.pathelement_[i].OutputPartial(out)
            out.putVarInt32(116)
        if self.has_name_space_:
            out.putVarInt32(162)
            out.putPrefixedString(self.name_space_)
        if self.has_database_id_:
            out.putVarInt32(186)
            out.putPrefixedString(self.database_id_)

    def TryMerge(self, d):
        while 1:
            tt = d.getVarInt32()
            if tt == 100:
                break
            if tt == 106:
                self.set_app(d.getPrefixedString())
                continue
            if tt == 115:
                self.add_pathelement().TryMerge(d)
                continue
            if tt == 162:
                self.set_name_space(d.getPrefixedString())
                continue
            if tt == 186:
                self.set_database_id(d.getPrefixedString())
                continue
            if tt == 0:
                raise ProtocolBuffer.ProtocolBufferDecodeError()
            d.skipData(tt)

    def __str__(self, prefix='', printElemNumber=0):
        res = ''
        if self.has_app_:
            res += prefix + 'app: %s\n' % self.DebugFormatString(self.app_)
        if self.has_name_space_:
            res += prefix + 'name_space: %s\n' % self.DebugFormatString(self.name_space_)
        cnt = 0
        for e in self.pathelement_:
            elm = ''
            if printElemNumber:
                elm = '(%d)' % cnt
            res += prefix + 'PathElement%s {\n' % elm
            res += e.__str__(prefix + '  ', printElemNumber)
            res += prefix + '}\n'
            cnt += 1
        if self.has_database_id_:
            res += prefix + 'database_id: %s\n' % self.DebugFormatString(self.database_id_)
        return res