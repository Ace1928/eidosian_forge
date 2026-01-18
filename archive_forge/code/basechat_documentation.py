from twisted.words.im.locals import AWAY, OFFLINE, ONLINE

        For the given C{person}, change the C{person}'s C{name} to C{newnick}
        and tell the contact list and any conversation windows with that
        C{person} to change as well.

        @type person: L{IPerson<interfaces.IPerson>} provider
        @param person: The person whose nickname will get changed.

        @type newnick: C{str}
        @param newnick: The new C{name} C{person} will take.
        