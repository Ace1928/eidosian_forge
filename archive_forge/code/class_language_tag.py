class language_tag(object):
    """This class represents an RFC 3066 language tag.

    Initialize objects of this class with a single string representing
    the language tag, such as "en-US".
        
    Case is insensitive. Wildcarded subtags are ignored or stripped as
    they have no significance, so that "en-*" is the same as "en".
    However the universal wildcard "*" language tag is kept as-is.

    Note that although relational operators such as < are defined,
    they only form a partial order based upon specialization.

    Thus for example,
         "en" <= "en-US"
    but,
         not "en" <= "de", and
         not "de" <= "en".

    """

    def __init__(self, tagname):
        """Initialize objects of this class with a single string representing
        the language tag, such as "en-US".  Case is insensitive.

        """
        self.parts = tagname.lower().split('-')
        while len(self.parts) > 1 and self.parts[-1] == '*':
            del self.parts[-1]

    def __len__(self):
        """Number of subtags in this tag."""
        if len(self.parts) == 1 and self.parts[0] == '*':
            return 0
        return len(self.parts)

    def __str__(self):
        """The standard string form of this language tag."""
        a = []
        if len(self.parts) >= 1:
            a.append(self.parts[0])
        if len(self.parts) >= 2:
            if len(self.parts[1]) == 2:
                a.append(self.parts[1].upper())
            else:
                a.append(self.parts[1])
        a.extend(self.parts[2:])
        return '-'.join(a)

    def __unicode__(self):
        """The unicode string form of this language tag."""
        return str(self.__str__())

    def __repr__(self):
        """The python representation of this language tag."""
        s = '%s("%s")' % (self.__class__.__name__, self.__str__())
        return s

    def superior(self):
        """Returns another instance of language_tag which is the superior.

        Thus en-US gives en, and en gives *.

        """
        if len(self) <= 1:
            return self.__class__('*')
        return self.__class__('-'.join(self.parts[:-1]))

    def all_superiors(self, include_wildcard=False):
        """Returns a list of this language and all it's superiors.

        If include_wildcard is False, then "*" will not be among the
        output list, unless this language is itself "*".

        """
        langlist = [self]
        l = self
        while not l.is_universal_wildcard():
            l = l.superior()
            if l.is_universal_wildcard() and (not include_wildcard):
                continue
            langlist.append(l)
        return langlist

    def is_universal_wildcard(self):
        """Returns True if this language tag represents all possible
        languages, by using the reserved tag of "*".

        """
        return len(self.parts) == 1 and self.parts[0] == '*'

    def dialect_of(self, other, ignore_wildcard=True):
        """Is this language a dialect (or subset/specialization) of another.

        This method returns True if this language is the same as or a
        specialization (dialect) of the other language_tag.

        If ignore_wildcard is False, then all languages will be
        considered to be a dialect of the special language tag of "*".

        """
        if not ignore_wildcard and self.is_universal_wildcard():
            return True
        for i in range(min(len(self), len(other))):
            if self.parts[i] != other.parts[i]:
                return False
        if len(self) >= len(other):
            return True
        return False

    def __eq__(self, other):
        """== operator. Are the two languages the same?"""
        return self.parts == other.parts

    def __neq__(self, other):
        """!= operator. Are the two languages different?"""
        return not self.__eq__(other)

    def __lt__(self, other):
        """< operator. Returns True if the other language is a more
        specialized dialect of this one."""
        return other.dialect_of(self) and self != other

    def __le__(self, other):
        """<= operator. Returns True if the other language is the same
        as or a more specialized dialect of this one."""
        return other.dialect_of(self)

    def __gt__(self, other):
        """> operator.  Returns True if this language is a more
        specialized dialect of the other one."""
        return self.dialect_of(other) and self != other

    def __ge__(self, other):
        """>= operator.  Returns True if this language is the same as
        or a more specialized dialect of the other one."""
        return self.dialect_of(other)