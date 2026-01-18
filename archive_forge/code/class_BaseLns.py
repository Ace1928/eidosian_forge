from sys import version_info as _swig_python_version_info
import weakref
class BaseLns(IntVarLocalSearchOperator):
    """
    This is the base class for building an Lns operator. An Lns fragment is a
    collection of variables which will be relaxed. Fragments are built with
    NextFragment(), which returns false if there are no more fragments to build.
    Optionally one can override InitFragments, which is called from
    LocalSearchOperator::Start to initialize fragment data.

    Here's a sample relaxing one variable at a time:

    class OneVarLns : public BaseLns {
     public:
      OneVarLns(const std::vector<IntVar*>& vars) : BaseLns(vars), index_(0) {}
      virtual ~OneVarLns() {}
      virtual void InitFragments() { index_ = 0; }
      virtual bool NextFragment() {
        const int size = Size();
        if (index_ < size) {
          AppendToFragment(index_);
          ++index_;
          return true;
        } else {
          return false;
        }
      }

     private:
      int index_;
    };
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, vars):
        if self.__class__ == BaseLns:
            _self = None
        else:
            _self = self
        _pywrapcp.BaseLns_swiginit(self, _pywrapcp.new_BaseLns(_self, vars))
    __swig_destroy__ = _pywrapcp.delete_BaseLns

    def InitFragments(self):
        return _pywrapcp.BaseLns_InitFragments(self)

    def NextFragment(self):
        return _pywrapcp.BaseLns_NextFragment(self)

    def AppendToFragment(self, index):
        return _pywrapcp.BaseLns_AppendToFragment(self, index)

    def FragmentSize(self):
        return _pywrapcp.BaseLns_FragmentSize(self)

    def __getitem__(self, index):
        return _pywrapcp.BaseLns___getitem__(self, index)

    def __len__(self):
        return _pywrapcp.BaseLns___len__(self)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_BaseLns(self)
        return weakref.proxy(self)