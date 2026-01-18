import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
class GroupHomomorphism:
    """
    A class representing group homomorphisms. Instantiate using `homomorphism()`.

    References
    ==========

    .. [1] Holt, D., Eick, B. and O'Brien, E. (2005). Handbook of computational group theory.

    """

    def __init__(self, domain, codomain, images):
        self.domain = domain
        self.codomain = codomain
        self.images = images
        self._inverses = None
        self._kernel = None
        self._image = None

    def _invs(self):
        """
        Return a dictionary with `{gen: inverse}` where `gen` is a rewriting
        generator of `codomain` (e.g. strong generator for permutation groups)
        and `inverse` is an element of its preimage

        """
        image = self.image()
        inverses = {}
        for k in list(self.images.keys()):
            v = self.images[k]
            if not (v in inverses or v.is_identity):
                inverses[v] = k
        if isinstance(self.codomain, PermutationGroup):
            gens = image.strong_gens
        else:
            gens = image.generators
        for g in gens:
            if g in inverses or g.is_identity:
                continue
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                parts = image._strong_gens_slp[g][::-1]
            else:
                parts = g
            for s in parts:
                if s in inverses:
                    w = w * inverses[s]
                else:
                    w = w * inverses[s ** (-1)] ** (-1)
            inverses[g] = w
        return inverses

    def invert(self, g):
        """
        Return an element of the preimage of ``g`` or of each element
        of ``g`` if ``g`` is a list.

        Explanation
        ===========

        If the codomain is an FpGroup, the inverse for equal
        elements might not always be the same unless the FpGroup's
        rewriting system is confluent. However, making a system
        confluent can be time-consuming. If it's important, try
        `self.codomain.make_confluent()` first.

        """
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.free_groups import FreeGroupElement
        if isinstance(g, (Permutation, FreeGroupElement)):
            if isinstance(self.codomain, FpGroup):
                g = self.codomain.reduce(g)
            if self._inverses is None:
                self._inverses = self._invs()
            image = self.image()
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                gens = image.generator_product(g)[::-1]
            else:
                gens = g
            for i in range(len(gens)):
                s = gens[i]
                if s.is_identity:
                    continue
                if s in self._inverses:
                    w = w * self._inverses[s]
                else:
                    w = w * self._inverses[s ** (-1)] ** (-1)
            return w
        elif isinstance(g, list):
            return [self.invert(e) for e in g]

    def kernel(self):
        """
        Compute the kernel of `self`.

        """
        if self._kernel is None:
            self._kernel = self._compute_kernel()
        return self._kernel

    def _compute_kernel(self):
        G = self.domain
        G_order = G.order()
        if G_order is S.Infinity:
            raise NotImplementedError('Kernel computation is not implemented for infinite groups')
        gens = []
        if isinstance(G, PermutationGroup):
            K = PermutationGroup(G.identity)
        else:
            K = FpSubgroup(G, gens, normal=True)
        i = self.image().order()
        while K.order() * i != G_order:
            r = G.random()
            k = r * self.invert(self(r)) ** (-1)
            if k not in K:
                gens.append(k)
                if isinstance(G, PermutationGroup):
                    K = PermutationGroup(gens)
                else:
                    K = FpSubgroup(G, gens, normal=True)
        return K

    def image(self):
        """
        Compute the image of `self`.

        """
        if self._image is None:
            values = list(set(self.images.values()))
            if isinstance(self.codomain, PermutationGroup):
                self._image = self.codomain.subgroup(values)
            else:
                self._image = FpSubgroup(self.codomain, values)
        return self._image

    def _apply(self, elem):
        """
        Apply `self` to `elem`.

        """
        if elem not in self.domain:
            if isinstance(elem, (list, tuple)):
                return [self._apply(e) for e in elem]
            raise ValueError('The supplied element does not belong to the domain')
        if elem.is_identity:
            return self.codomain.identity
        else:
            images = self.images
            value = self.codomain.identity
            if isinstance(self.domain, PermutationGroup):
                gens = self.domain.generator_product(elem, original=True)
                for g in gens:
                    if g in self.images:
                        value = images[g] * value
                    else:
                        value = images[g ** (-1)] ** (-1) * value
            else:
                i = 0
                for _, p in elem.array_form:
                    if p < 0:
                        g = elem[i] ** (-1)
                    else:
                        g = elem[i]
                    value = value * images[g] ** p
                    i += abs(p)
        return value

    def __call__(self, elem):
        return self._apply(elem)

    def is_injective(self):
        """
        Check if the homomorphism is injective

        """
        return self.kernel().order() == 1

    def is_surjective(self):
        """
        Check if the homomorphism is surjective

        """
        im = self.image().order()
        oth = self.codomain.order()
        if im is S.Infinity and oth is S.Infinity:
            return None
        else:
            return im == oth

    def is_isomorphism(self):
        """
        Check if `self` is an isomorphism.

        """
        return self.is_injective() and self.is_surjective()

    def is_trivial(self):
        """
        Check is `self` is a trivial homomorphism, i.e. all elements
        are mapped to the identity.

        """
        return self.image().order() == 1

    def compose(self, other):
        """
        Return the composition of `self` and `other`, i.e.
        the homomorphism phi such that for all g in the domain
        of `other`, phi(g) = self(other(g))

        """
        if not other.image().is_subgroup(self.domain):
            raise ValueError('The image of `other` must be a subgroup of the domain of `self`')
        images = {g: self(other(g)) for g in other.images}
        return GroupHomomorphism(other.domain, self.codomain, images)

    def restrict_to(self, H):
        """
        Return the restriction of the homomorphism to the subgroup `H`
        of the domain.

        """
        if not isinstance(H, PermutationGroup) or not H.is_subgroup(self.domain):
            raise ValueError('Given H is not a subgroup of the domain')
        domain = H
        images = {g: self(g) for g in H.generators}
        return GroupHomomorphism(domain, self.codomain, images)

    def invert_subgroup(self, H):
        """
        Return the subgroup of the domain that is the inverse image
        of the subgroup ``H`` of the homomorphism image

        """
        if not H.is_subgroup(self.image()):
            raise ValueError('Given H is not a subgroup of the image')
        gens = []
        P = PermutationGroup(self.image().identity)
        for h in H.generators:
            h_i = self.invert(h)
            if h_i not in P:
                gens.append(h_i)
                P = PermutationGroup(gens)
            for k in self.kernel().generators:
                if k * h_i not in P:
                    gens.append(k * h_i)
                    P = PermutationGroup(gens)
        return P