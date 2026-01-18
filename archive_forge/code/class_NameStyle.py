from __future__ import unicode_literals
from pybtex.style.names import BaseNameStyle, name_part
from pybtex.style.template import join
class NameStyle(BaseNameStyle):

    def format(self, person, abbr=False):
        """
        Format names similarly to {ff~}{vv~}{ll}{, jj} in BibTeX.

        >>> from pybtex.database import Person
        >>> name = Person(string=r"Charles Louis Xavier Joseph de la Vall{\\'e}e Poussin")
        >>> plain = NameStyle().format

        >>> print(plain(name).format().render_as('latex'))
        Charles Louis Xavier~Joseph de~la Vall{é}e~Poussin
        >>> print(plain(name).format().render_as('html'))
        Charles Louis Xavier&nbsp;Joseph de&nbsp;la Vall<span class="bibtex-protected">é</span>e&nbsp;Poussin

        >>> print(plain(name, abbr=True).format().render_as('latex'))
        C.~L. X.~J. de~la Vall{é}e~Poussin
        >>> print(plain(name, abbr=True).format().render_as('html'))
        C.&nbsp;L. X.&nbsp;J. de&nbsp;la Vall<span class="bibtex-protected">é</span>e&nbsp;Poussin

        >>> name = Person(first='First', last='Last', middle='Middle')
        >>> print(plain(name).format().render_as('latex'))
        First~Middle Last

        >>> print(plain(name, abbr=True).format().render_as('latex'))
        F.~M. Last

        >>> print(plain(Person('de Last, Jr., First Middle')).format().render_as('latex'))
        First~Middle de~Last, Jr.

        """
        return join[name_part(tie=True, abbr=abbr)[person.rich_first_names + person.rich_middle_names], name_part(tie=True)[person.rich_prelast_names], name_part[person.rich_last_names], name_part(before=', ')[person.rich_lineage_names]]