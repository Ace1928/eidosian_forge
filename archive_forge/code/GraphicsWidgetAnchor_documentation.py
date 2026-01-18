from ..Point import Point

        Set the position of this item relative to its parent by automatically 
        choosing appropriate anchor settings.
        
        If relative is True, one corner of the item will be anchored to 
        the appropriate location on the parent with no offset. The anchored
        corner will be whichever is closest to the parent's boundary.
        
        If relative is False, one corner of the item will be anchored to the same
        corner of the parent, with an absolute offset to achieve the correct
        position. 
        