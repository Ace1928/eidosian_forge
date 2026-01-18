import pygame as pg
import pygame.camera
class VideoCapturePlayer:
    size = (640, 480)

    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super().__init__(**argd)
        self.display = pg.display.set_mode(self.size)
        self.init_cams(0)

    def init_cams(self, which_cam_idx):
        self.clist = pygame.camera.list_cameras()
        if not self.clist:
            raise ValueError('Sorry, no cameras detected.')
        try:
            cam_id = self.clist[which_cam_idx]
        except IndexError:
            cam_id = self.clist[0]
        self.camera = pygame.camera.Camera(cam_id, self.size, 'RGB')
        self.camera.start()
        self.clock = pg.time.Clock()
        self.snapshot = pg.surface.Surface(self.size, 0, self.display)
        return cam_id

    def get_and_flip(self):
        self.snapshot = self.camera.get_image(self.display)
        pg.display.flip()

    def main(self):
        clist = pygame.camera.list_cameras()
        if not clist:
            raise ValueError('Sorry, no cameras detected.')
        camera = clist[0]
        print('\nPress the associated number for the desired camera to see that display!')
        print('(Selecting a camera that does not exist will default to camera 0)')
        for index, cam in enumerate(clist):
            print(f'[{index}]: {cam}')
        going = True
        while going:
            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    going = False
                if e.type == pg.KEYDOWN:
                    if e.key in range(pg.K_0, pg.K_0 + 10):
                        camera = self.init_cams(e.key - pg.K_0)
            self.get_and_flip()
            self.clock.tick()
            pygame.display.set_caption(f'{camera} ({self.clock.get_fps():.2f} FPS)')